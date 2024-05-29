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
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionPipeline
import logging

device = "cuda:7" if torch.cuda.is_available() else "cpu"

_tokenizer = _Tokenizer()
# df = pd.DataFrame()
# cls_label = pd.DataFrame()


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
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class DomainClassPL(nn.Module):
    def __init__(self, classnames, domainnames, clip_model):
        super().__init__()

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        n_cls = len(classnames)

        ctx_init1 = "a"
        ctx_init2 = "of a"
        n_ctx = 4

        domainnames = [domain.replace("_", " ") for domain in domainnames]
        ctx_list = [ctx_init1 + " " + domain + " " + ctx_init2 for domain in domainnames]

        ctx_init = [ctx.replace("_", " ") for ctx in ctx_list]
        prompt = clip.tokenize(ctx_init)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        prompt_prefixes = ctx_init

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = []
        for prefix in prompt_prefixes:
            class_prompts = [prefix + " " + name + "." for name in classnames]
            prompts.extend(class_prompts)

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS


        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.n_dm = len(prompts)
        self.tokenized_prompts = tokenized_prompts  
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_dm, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = ctx.to(device)

        prompts = torch.cat(
            [
                prefix,  # (n_dm, 1, dim)
                ctx,     # (n_dm, n_ctx, dim)
                suffix,  # (n_dm, *, dim)
            ],
            dim=1,
        )
        return prompts
    
class DomainPL(nn.Module):
    def __init__(self, classnames, domainnames, clip_model):
        super().__init__()

        n_cls = len(classnames)
        n_dmn = len(domainnames)
        n_ctx = 4
        ctx_init = "this image belongs to"
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        ctx_init = ctx_init.replace("_", " ")
        n_ctx = len(ctx_init.split(" "))
        prompt = clip.tokenize(ctx_init)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        prompt_prefix = ctx_init

        self.ctx = nn.Parameter(ctx_vectors)  

        domainnames = [domain.replace("_", " ") for domain in domainnames]
        domain_lens = [len(_tokenizer.encode(domain)) for domain in domainnames]
        domain_texts = [prompt_prefix + " " + domain + "." for domain in domainnames]
        prompts = [x for x in domain_texts for _ in range(n_cls)]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_dmn = len(prompts)
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.domain_lens = domain_lens
        self.projector = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(ctx_dim, ctx_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(ctx_dim // 16, ctx_dim))
        ]))

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_dmn, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_dmn, 1, dim)
                ctx,     # (n_dmn, n_ctx, dim)
                suffix,  # (n_dmn, *, dim)
            ],
            dim=1,
        )
        return prompts

    
class UnknownPL(nn.Module):
    def __init__(self, clip_model):
        super().__init__()

        n_ctx = 4
        ctx_init = "an image of a"
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, mean=0.5, std=1.0)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors) 

        classnames = ['unspecified']
        n_cls = len(classnames)
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        # self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION 

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim) 
            ],
            dim=1,
        )

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
    
# model_id_or_path = "runwayml/stable-diffusion-v1-5"
# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
# pipe = pipe.to(device) 

# class StableDiffusion(nn.Module):
#     def __init__(self):
#         super(StableDiffusion, self).__init__()
#         self.model_id_or_path = "runwayml/stable-diffusion-v1-5"
#         self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.model_id_or_path, torch_dtype=torch.float16)
#         self.pipe = self.pipe.to(device)
#     def forward(self, images , pos_prompt, neg_prompt):
#         generated_images = []
#         images = images.to(device)
#         images_normalized = (images - images.min()) / (images.max() - images.min())
#         images_normalized = images_normalized.to(device)

#         for x in tqdm(images_normalized, leave=False):
#             with torch.no_grad():
#                 batch_output = self.pipe(prompt=pos_prompt, negative_prompt=neg_prompt, image=x.unsqueeze(0), strength=0.9, guidance_scale=15)

#             generated_images.append(batch_output.images[0])
        
#         generated_images = torch.stack([ToTensor()(img) for img in generated_images]).to(device)
#         return generated_images

class StableDiffusion(nn.Module):
    def __init__(self):
        super(StableDiffusion, self).__init__()
        self.model_id_or_path = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id_or_path, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(device)
        logging.basicConfig(level=logging.WARNING)
    def forward(self, images, pos_prompt, neg_prompt):       
        generated_images = []
        batchsize = images.shape[0] * 0.25
        batchsize = int(batchsize)
        
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

norm_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
  

class GenerateUnknownImages(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.unknown_prompt_learner = UnknownPL(clip_model) 
        self.unknown_tokenized_prompts = self.unknown_prompt_learner.tokenized_prompts 
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model) 
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.diffusion = StableDiffusion()

    def forward(self, image, pos_prompt, neg_prompt):
        # unknown_prompt = self.unknown_prompt_learner()
        # unknown_tokenized_prompts = self.unknown_tokenized_prompts  

        # unknown_text_features = self.text_encoder(unknown_prompt, unknown_tokenized_prompts)
        # unknown_text_features = unknown_text_features / unknown_text_features.norm(dim=-1, keepdim=True)  
        # unknown_text_features1 = unknown_text_features.unsqueeze(2).repeat(1, 1, 512)
        # matched_texts = unknown_text_features1.repeat(len(image),3,1,1)
        # matched_texts = matched_texts.to(device)

        '''
        Stable diffusion
        '''

        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]      
        normalize = transforms.Normalize(mean=mean, std=std)
        
        generated_unknown_images = self.diffusion(image, pos_prompt, neg_prompt) 
        resized_unknown_images = torch.stack([resize_transform(x) for x in generated_unknown_images])
        generated_unknown_images1 = normalize(resized_unknown_images)
        generated_unknown_images1 = generated_unknown_images1.to(device)

        return generated_unknown_images, generated_unknown_images1


class CustomCLIP(nn.Module):
    def __init__(self, classnames, domainnames, clip_model):
        super().__init__()
        self.domainclass_pl = DomainClassPL(classnames, domainnames, clip_model)
        self.domainclass_tokenizedprompts = self.domainclass_pl.tokenized_prompts
        self.domain_pl = DomainPL(classnames, domainnames, clip_model)
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
        # global df 
        # global cls_label
        
        domainclass_prompts = self.domainclass_pl()   
        domainclass_tokenizedprompts = self.domainclass_tokenizedprompts
        domainclass_textfeatures = self.text_encoder(domainclass_prompts, domainclass_tokenizedprompts)
        domainclass_textfeatures = domainclass_textfeatures / domainclass_textfeatures.norm(dim=-1, keepdim=True)
        #print(domainclass_textfeatures.shape)
        

        domain_prompts = self.domain_pl()   
        domain_tokenizedprompts = self.domain_tokenizedprompts
        domain_textfeatures = self.text_encoder(domain_prompts, domain_tokenizedprompts)
        domain_textfeatures = domain_textfeatures / domain_textfeatures.norm(dim=-1, keepdim=True)
        #print(domain_textfeatures.shape)


        diff = domainclass_textfeatures - domain_textfeatures
        diff = diff / diff.norm(dim=-1, keepdim=True)
        l2_norm = torch.linalg.norm(diff, dim=1, ord=2)   
        l2_norm = l2_norm.unsqueeze(1)

        norm_textfeatures = diff / l2_norm
        norm_textfeatures = norm_textfeatures / norm_textfeatures.norm(dim=-1, keepdim=True)

        grouped_dfeat = []
        
        for i in range(self.num_class):
            group = norm_textfeatures[i::self.num_class]
            grouped_dfeat.append(group)
      
        reshaped_class_text_features = torch.stack(grouped_dfeat)
        averaged_class_text_features = torch.mean(reshaped_class_text_features, dim=1, keepdim=True)
        class_textfeatures = averaged_class_text_features.squeeze(1)
        # print(class_textfeatures.shape)
        # exit()
        
        values_tensor = []
        for value in label:
            classtext_value = class_textfeatures[value, :]
            values_tensor.append(classtext_value)
        matched_texts = torch.stack(values_tensor) 

        self.upsample_net = self.upsample_net.to(device)
        upsampled_texts = self.upsample_net(matched_texts)
        target_size = (224, 224)
        final_texts = F.interpolate(upsampled_texts, size=target_size, mode='bilinear', align_corners=False)
        
        new_image = torch.cat((image, final_texts), dim=1).to(device)
        final_image = self.conv_layer(new_image.to(torch.float32))

        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        
        normalize = transforms.Normalize(mean=mean, std=std)

        rescaled_image = normalize(final_image)
        # sigmoid_image = torch.sigmoid(rescaled_image)
        # attention_image = (sigmoid_image * image) + rescaled_image
        
        image_features = self.image_encoder(image.type(self.dtype))

        #image_features = img_features + matched_texts

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        
        #t = image_features.squeeze(1).cpu().detach().numpy()
        # t = image_features.cpu().detach().numpy()
        # label_np = label.cpu().detach().numpy()
        # df1 = pd.DataFrame(t)
        # df2 = pd.DataFrame(label_np)
        #print('df1 shape:',df1.shape)
        #df1[''] = label
        # df = df.append(df1, ignore_index=True)   
        # cls_label = cls_label.append(df2, ignore_index=True)    
        # df = pd.concat([df, df1], ignore_index=True)
        # cls_label = pd.concat([cls_label, df2], ignore_index=True) 
        # df.to_csv('/home/dgxadmin/Ankit/ODG/csv/pacs/sketch_img.csv', header=False, index=False) 
        # cls_label.to_csv('/home/dgxadmin/Ankit/ODG/csv/pacs/sketch_label.csv', header=False, index=False) 
        #print('df :',df.shape) 
        #print('cls_label :', cls_label.shape)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ class_textfeatures.t()
        cosine_sim1 = F.cosine_similarity(domainclass_textfeatures,domain_textfeatures , dim=0)
        diff_proj = self.textprojector(domainclass_textfeatures) - self.textprojector(domain_textfeatures)
        diff_proj = diff_proj / diff_proj.norm(dim=-1, keepdim=True)
        #print(class_textfeatures.shape,torch.mean(diff_proj,dim=1).shape)
        #print("cosine_sim1",torch.mean(cosine_sim1))
        l2_normproj = torch.linalg.norm(diff_proj, dim=1, ord=2)   
        l2_normproj = l2_normproj.unsqueeze(1)

        diff_projfeatures  = diff_proj / l2_normproj
        diff_projfeatures  = diff_projfeatures  / diff_projfeatures .norm(dim=-1, keepdim=True)
        class_textfeatures=torch.cat((class_textfeatures,class_textfeatures,class_textfeatures),dim=0)
        #print(diff_projfeatures.shape,class_textfeatures.shape)
        #exit()
        cosine_sim2=F.cosine_similarity(class_textfeatures,diff_projfeatures,dim=0)

        #print("cosine_sim2",torch.mean(cosine_sim2))
        return logits, diff_projfeatures 
