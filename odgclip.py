import os.path as osp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn import functional as F
import torchvision.transforms.functional as TF
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import torchvision.transforms as transforms
import os
from torchvision.models import resnet18, resnet50
from torchvision.transforms import ToTensor

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
# from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionPipeline
import logging

device = "cuda:0" if torch.cuda.is_available() else "cpu"

_tokenizer = _Tokenizer()
df = pd.DataFrame()
cls_label = pd.DataFrame()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()
    
    def mu(self, x):
        return torch.sum(x,(1))/(x.shape[1])
    
    def sigma(self, x):
        return torch.sqrt((torch.sum((x.permute([1,0,2])-self.mu(x)).permute([1,0,2])**2,(1))+0.000000023)/(x.shape[1]))


class style_projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.ModuleList(nn.Linear(768,256) for _ in range (12))
        self.linear2 = nn.ModuleList(nn.Linear(256,512) for _ in range (12))
        self.adain=AdaIN()
        self.gap=nn.AdaptiveAvgPool2d((1,768))
    def forward(self, data):
        data_prompt=[]
        for i in range(len(data)):
            x_mu=self.adain.mu(data[i]).unsqueeze(1).to(torch.float32)
            x_sigma=self.adain.sigma(data[i]).unsqueeze(1).to(torch.float32)
            x_cat = torch.cat((x_mu, x_sigma),1)
            x_cat = self.gap(x_cat).squeeze(1)
            x_out = self.linear1[i](x_cat)
            x_final = self.linear2[i](x_out)
            data_prompt.append(x_final)
        output = torch.stack(data_prompt, dim=1)
        return output
    

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts.to(device) + self.positional_embedding.type(self.dtype).to(device)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x,_ = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class DomainClassPL(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_imgctx = 4
        n_domainctx = 12
        n_ctx = n_imgctx + n_domainctx

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        ctx_vectors = torch.empty(n_imgctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        self.style = style_projector()
        self.frg = nn.AdaptiveAvgPool2d((1,vis_dim))
        self.rho = nn.Linear(12,1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) 
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :]) 
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts 
        self.name_lens = name_lens
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  
                ctx,   
                suffix, 
            ],
            dim=1,
        )

        return prompts
    
    def forward(self, data):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx  
       
        style_feat = self.style(data)
        new_ctx = ctx.repeat(style_feat.shape[0], 1, 1) 
        ctx_shifted = torch.cat((style_feat, new_ctx),1)   
        
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix) 
            prompts.append(pts_i)
        prompts = torch.stack(prompts)     
        return prompts
    
    
class DomainPL(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)  # here we will use only the number of classes, not class names
        n_ctx = 4
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["Y"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        self.style = style_projector()
        # self.frg = nn.AdaptiveAvgPool2d((1,vis_dim))
        # self.rho = nn.Linear(12,1)

        # domainnames = [name.replace("_", " ") for name in domainnames]
        # name_lens = [len(_tokenizer.encode(name)) for name in domainnames]
        # prompts = [prompt_prefix + " " + name + "." for name in domainnames]

        prompts = [prompt_prefix]*n_cls

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) 
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :]) 
        self.register_buffer("token_suffix", embedding[:, 1 + 12 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts 
        # self.name_lens = name_lens
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  
                ctx,   
                suffix, 
            ],
            dim=1,
        )

        return prompts
    
    def forward(self, data):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx          
    
        style_feat = self.style(data)
        
        new_ctx = ctx.repeat(style_feat.shape[0], 1, 1) 
        ctx_shifted = torch.cat((style_feat, new_ctx),1)   
        
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix) 
            prompts.append(pts_i)
        prompts = torch.stack(prompts)     
        return prompts

  

class ConvLayer(nn.Module):
    def __init__(self):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(4, 3, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UpsampleNetwork(nn.Module):
    def __init__(self):
        super(UpsampleNetwork, self).__init__()
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=7, stride=3, padding=1, output_padding=2)
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=7, stride=3, padding=1, output_padding=2)
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=7, stride=3, padding=1, output_padding=2)
        self.conv4 = nn.ConvTranspose2d(64, 1, kernel_size=7, stride=3, padding=1, output_padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x.unsqueeze(-1).unsqueeze(-1)))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)

        return x


class StableDiffusion(nn.Module):
    def __init__(self):
        super(StableDiffusion, self).__init__()
        self.model_id_or_path = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id_or_path, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(device)
        logging.basicConfig(level=logging.WARNING)
    def forward(self, batch, pos_prompt, neg_prompt):       
        generated_images = []
        # batchsize = images.shape[0] * 0.25
        batchsize = int(batch*0.25)
        
        positive_prompt = [pos_prompt] * batchsize
        negative_prompt = [neg_prompt] * batchsize

        with torch.no_grad():
            for i in range(batchsize):
                batch_output = self.pipe(prompt=positive_prompt[i], negative_prompt=negative_prompt[i], guidance_scale=15)
                generated_images.append(batch_output.images[0])
        generated_images = torch.stack([ToTensor()(img) for img in generated_images]).to(device)
        return generated_images
    
resize_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# norm_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((512, 512)),
#     transforms.ToTensor(),
# ])
  
class GenerateUnknownImages(nn.Module):
    def __init__(self):
        super().__init__()

        self.diffusion = StableDiffusion()

    def forward(self, batch, pos_prompt, neg_prompt):
        '''
        Stable diffusion
        '''

        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]      
        normalize = transforms.Normalize(mean=mean, std=std)
        
        generated_unknown_images = self.diffusion(batch, pos_prompt, neg_prompt) 
        resized_unknown_images = torch.stack([resize_transform(x) for x in generated_unknown_images])
        generated_unknown_images1 = normalize(resized_unknown_images)
        generated_unknown_images1 = generated_unknown_images1.to(device)

        return generated_unknown_images, generated_unknown_images1

class CustomCLIP(nn.Module):
    def __init__(self, classnames, domainnames, clip_model):
        super().__init__()
        self.domainclass_pl = DomainClassPL(classnames, clip_model)
        self.domainclass_tokenizedprompts = self.domainclass_pl.tokenized_prompts
        self.domain_pl = DomainPL(classnames, clip_model)
        self.domain_tokenizedprompts = self.domain_pl.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.conv_layer = ConvLayer()
        self.upsample_net = UpsampleNetwork()
        self.num_class = len(classnames)
        self.num_domain = len(domainnames)
        self.textprojector = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(512, 512 // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(512 // 16, 512))
        ]))


    def forward(self, image, label):
        global df 
        global cls_label

        raw_imgfeat, data = self.image_encoder(image.type(self.dtype))
        
        domainclass_prompts = self.domainclass_pl(data) 
        domainclass_tokenizedprompts = self.domainclass_tokenizedprompts

        domainclass_textfeatures = []
        for pts_i in domainclass_prompts:
            text_features = self.text_encoder(pts_i, domainclass_tokenizedprompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            domainclass_textfeatures.append(text_features)
        domainclass_textfeatures = torch.stack(domainclass_textfeatures)
        # domainclass_textfeatures = domainclass_textfeatures / domainclass_textfeatures.norm(dim=-1, keepdim=True)
        

        domain_prompts = self.domain_pl(data)   
        domain_tokenizedprompts = self.domain_tokenizedprompts

        domain_textfeatures = []
        for pts_i in domain_prompts:
            text_features = self.text_encoder(pts_i, domain_tokenizedprompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            domain_textfeatures.append(text_features)
        domain_textfeatures = torch.stack(domain_textfeatures)
        # domain_textfeatures = domain_textfeatures / domain_textfeatures.norm(dim=-1, keepdim=True)

        diff = domainclass_textfeatures - domain_textfeatures
        diff = diff / diff.norm(dim=-1, keepdim=True)
        # l2_norm = torch.linalg.norm(diff, dim=-1, ord=2)   
        # l2_norm = l2_norm.unsqueeze(2)

        # norm_textfeatures = diff / l2_norm
        # norm_textfeatures = norm_textfeatures / norm_textfeatures.norm(dim=-1, keepdim=True)

        # grouped_dfeat = []
        
        # for i in range(self.num_class):
        #     group = norm_textfeatures[i::self.num_class]
        #     print(group.shape)
        #     exit()
        #     grouped_dfeat.append(group)
      
        # reshaped_class_text_features = torch.stack(grouped_dfeat)
        # averaged_class_text_features = torch.mean(reshaped_class_text_features, dim=1, keepdim=True)
        # class_textfeatures = averaged_class_text_features.squeeze(1)

        # print(class_textfeatures.shape)
        # exit()
        
        # values_tensor = []
        # for value in label:
        #     classtext_value = norm_textfeatures[:, value, :]
        #     values_tensor.append(classtext_value)
        # matched_texts = torch.stack(values_tensor)

        matched_texts = torch.mean(diff, dim=1)

        self.upsample_net = self.upsample_net.to(device)
        upsampled_texts = self.upsample_net(matched_texts)
        target_size = (224, 224)
        final_texts = F.interpolate(upsampled_texts, size=target_size, mode='bilinear', align_corners=False)

        # '''
        # Saved x_tilde images
        # '''
        # generated_output_directory5 = '/home/dgxadmin/Ankit/ODG/generated_images/pacs/sketch_xtilde/x_tilde_images'
        # if not os.path.exists(generated_output_directory5):
        #    os.makedirs(generated_output_directory5)
        
        # generated_images_pil5 = [TF.to_pil_image(img) for img in final_texts]
        # for i, img_pil in enumerate(generated_images_pil5):
        #    img_pil.save(os.path.join(generated_output_directory5, f"xtilde_image_{i}.jpg"))
    
        new_image = torch.cat((image, final_texts), dim=1).to(device)
        final_image = self.conv_layer(new_image.to(torch.float32))


        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        
        normalize = transforms.Normalize(mean=mean, std=std)

        rescaled_image = normalize(final_image)

        # '''
        # Saved normalized latent images
        # '''
        # generated_output_directory6 = '/home/1098363-z100/nfs/users/Singha/testodg/test_sketch/normlatent_images'
        # if not os.path.exists(generated_output_directory6):
        #    os.makedirs(generated_output_directory6)
        
        # generated_images_pil6 = [TF.to_pil_image(img) for img in final_image]
        # for i, img_pil in enumerate(generated_images_pil6):
        #    img_pil.save(os.path.join(generated_output_directory6, f"normlatent_image_{i}.jpg"))

        # sigmoid_image = torch.sigmoid(rescaled_image)
        # attention_image = (sigmoid_image * image) + rescaled_image

        # '''
        # Saved x_tilde normalized images
        # '''
        # generated_output_directory6 = '/home/dgxadmin/Ankit/ODG/generated_images/pacs/sketch_xtilde/x_tilde_normalizedimages'
        # if not os.path.exists(generated_output_directory6):
        #    os.makedirs(generated_output_directory6)
        
        # generated_images_pil6 = [TF.to_pil_image(img) for img in final_image]
        # for i, img_pil in enumerate(generated_images_pil6):
        #    img_pil.save(os.path.join(generated_output_directory6, f"xtilde_normalizedimage_{i}.jpg"))
        
        image_features, _ = self.image_encoder(rescaled_image.type(self.dtype))

        #image_features = self.image_encoder(image.type(self.dtype))


        #image_features = img_features + matched_texts

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # raw_imgfeat = raw_imgfeat / raw_imgfeat.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ class_textfeatures.t()
        logits = []
        for pts_i, imf_i in zip(domainclass_textfeatures, image_features):
            pts_i = pts_i / pts_i.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ pts_i.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        # diff_proj = self.textprojector(domainclass_textfeatures) - self.textprojector(domain_textfeatures)
        # diff_proj = diff_proj / diff_proj.norm(dim=-1, keepdim=True)
        # l2_normproj = torch.linalg.norm(diff_proj, dim=-1, ord=2)   
        # l2_normproj = l2_normproj.unsqueeze(1)

        # diff_projfeatures  = diff_proj / l2_normproj
        # diff_projfeatures  = diff_projfeatures  / diff_projfeatures .norm(dim=-1, keepdim=True)

        #  # t = image_features.squeeze(1).cpu().detach().numpy()
        # # t = image_features.cpu().detach().numpy()
        # # t = class_textfeatures.squeeze(1).cpu().detach().numpy()
        # # t = class_textfeatures.cpu().detach().numpy()
        # # t = class_textfeatures.squeeze(1).cpu().detach().numpy()
        # t = logits.cpu().detach().numpy()
        # # print(label)
        # # print(label.shape)
        # # exit()
        # # text_label = torch.tensor([0, 1, 2, 3, 4, 5, 6])

        # label_np = label.cpu().detach().numpy()
        # df1 = pd.DataFrame(t)
        # df2 = pd.DataFrame(label_np)
        # # print('df1 shape:',df1.shape)
        # # df1[''] = label
        # # df = df.append(df1, ignore_index=True)   
        # # cls_label = cls_label.append(df2, ignore_index=True)    
        # df = pd.concat([df, df1], ignore_index=True)
        # cls_label = pd.concat([cls_label, df2], ignore_index=True) 
        # df.to_csv('/home/dgxadmin/Ankit/ODG/csv/pacs/photo_logits.csv', header=False, index=False) 
        # cls_label.to_csv('/home/dgxadmin/Ankit/ODG/csv/pacs/photo_logitslabel.csv', header=False, index=False) 
        # # print('df :',df.shape) 
        # # print('cls_label :', cls_label.shape)


        return logits, domainclass_textfeatures, final_image
        # return logits
    
    
