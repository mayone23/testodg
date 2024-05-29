import imp
import os.path as osp
from collections import OrderedDict
import math
import pandas as pd
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.modules.loss import _Loss

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
df = pd.DataFrame()
cls_label = pd.DataFrame()
# device = "cuda" if torch.cuda.is_available() else "cpu"
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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


class AdaIN_trans(nn.Module):
		def __init__(self):
				super().__init__()

		def mu(self, x):
				# print(x.shape)
				# exit()
				""" Takes a (n,c,h,w) tensor as input and returns the average across
				it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
				return torch.sum(x,(1))/(x.shape[1])

		def sigma(self, x):
				""" Takes a (n,c,h,w) tensor as input and returns the standard deviation
				across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
				the permutations are required for broadcasting"""
				return torch.sqrt((torch.sum((x.permute([1,0,2])-self.mu(x)).permute([1,0,2])**2,(1))+0.000000023)/(x.shape[1]))

		def forward(self, x, y):
				""" Takes a content embeding x and a style embeding y and changes
				transforms the mean and standard deviation of the content embedding to
				that of the style. [See eq. 8 of paper] Note the permutations are
				required for broadcasting"""
				return (self.sigma(y)*((x.permute([1,0,2])-self.mu(x))/self.sigma(x)) + self.mu(y)).permute([1,0,2])


# for ViT :
class multi_scale(nn.Module):
	def __init__(self):
		super(multi_scale,self).__init__()
		self.linear = nn.ModuleList(nn.Linear(768,512) for _ in range (12))
		self.adain=AdaIN_trans()
		self.gap=nn.AdaptiveAvgPool2d((1,1))
	def forward(self,data):
		data_prompt = []
		for i in range(len(data)):
			x_mu=self.adain.mu(data[i])
			x_mu = x_mu.to(torch.float32)
			x=self.linear[i](x_mu)
			data_prompt.append(x)
		data_prompt=torch.stack(data_prompt,1)  
		return data_prompt


class projector(nn.Module):
    def __init__(self):
        super(projector,self).__init__()
        self.adain=AdaIN_trans()
    def forward(self,im_features):
        im_prompt = []
        x_mu=self.adain.mu(im_features)
        im_prompt.append(x_mu)
        im_prompt=torch.stack(im_prompt,1) 
        return im_prompt
        

class InjectionBlock(nn.Module):
    def __init__(self, vis, ctx):
        super(InjectionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(vis, vis // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(vis // 16, vis, bias=False),
            nn.Sigmoid()
        )

        self.linearlayer1 = nn.Sequential(nn.Linear((vis*2),vis))
        self.linearlayer2 = nn.Sequential(nn.Linear((vis*2),ctx))
        self.gap=nn.AdaptiveAvgPool2d((1,512))
        #self.gap=nn.AdaptiveAvgPool2d(1)

    def forward(self, vis):  

        # x = adaptive_avg_pool(x)
        vis_f = self.gap(vis)
        # print(vis_f.shape)
        #vis_f = self.linearlayer1(vis.type(torch.float))  
        #print(f'vis_f : {vis_f.shape}')

        attn1 = self.attention(vis_f.type(torch.float))
        #print(f'attn1 : {attn1.shape}') 
        mulattn1 = torch.mul(attn1, vis_f)
        #print(f'mulattn1 : {mulattn1.shape}')
        resattn1 = torch.cat((mulattn1, vis_f),2)
        #print(f'resattn1 : {resattn1.shape}')
        linear1 = self.linearlayer1(resattn1)
        #print(f'linear1 : {linear1.shape}')

        attn2 = self.attention(linear1.type(torch.float))
        mulattn2 = torch.mul(attn2, vis_f)
        resattn2 = torch.cat((mulattn2, vis_f),2)
        linear2 = self.linearlayer2(resattn2)
        
        output = linear2.to(torch.float16)
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
		x = prompts + self.positional_embedding.type(self.dtype)
		#x = x.to(torch.float16)
		x = x.permute(1, 0, 2)  # NLD -> LND
		x,_ = self.transformer(x)
		x = x.permute(1, 0, 2)  # LND -> NLD
		x = self.ln_final(x).type(self.dtype)

		# x.shape = [batch_size, n_ctx, transformer.width]
		# take features from the eot embedding (eot_token is the highest number in each sequence)
		x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

		return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 4
        ctx_init = "a photo of a"
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = len(ctx_init.split(" "))
        prompt = clip.tokenize(ctx_init)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        prompt_prefix = ctx_init

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        print(f'Input size of attn: "{vis_dim}"')

        self.injection = InjectionBlock(vis_dim, ctx_dim)

        # self.rblock = rblock()

        # self.nnet = nnet()

        # self.style = style_prompt()

        self.multi = multi_scale()

        self.projector = projector()

        # self.linearlayer = nn.Sequential(nn.Linear(vis_dim,(vis_dim*4)))

        self.meta_net = nn.Sequential(OrderedDict([
			("linear1", nn.Linear(vis_dim, vis_dim // 16)),
			("relu", nn.ReLU(inplace=True)),
			("linear2", nn.Linear(vis_dim // 16, vis_dim))
		]))

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        #self.class_token_position = cfg.TRAINER.APPLENET.CLASS_TOKEN_POSITION 
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        
        # end position
        # if self.class_token_position == "end":
        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        
        # middle position
        # elif self.class_token_position == "middle":

        # half_n_ctx = self.n_ctx // 2
        # prompts = []
        # for i in range(self.n_cls):
        #     name_len = self.name_lens[i]
        #     prefix_i = prefix[i : i + 1, :, :]
        #     class_i = suffix[i : i + 1, :name_len, :]
        #     suffix_i = suffix[i : i + 1, name_len:, :]
        #     ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
        #     ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
        #     prompt = torch.cat(
        #         [
        #             prefix_i,     # (1, 1, dim)
        #             ctx_i_half1,  # (1, n_ctx//2, dim)
        #             class_i,      # (1, name_len, dim)
        #             ctx_i_half2,  # (1, n_ctx//2, dim)
        #             suffix_i,     # (1, *, dim)
        #         ],
        #         dim=1,
        #     )
        #     prompts.append(prompt)
        # prompts = torch.cat(prompts, dim=0)
        

        # front position
        #elif self.class_token_position == "front":

        # prompts = []
        # for i in range(self.n_cls):
        #     name_len = self.name_lens[i]
        #     prefix_i = prefix[i : i + 1, :, :]
        #     class_i = suffix[i : i + 1, :name_len, :]
        #     suffix_i = suffix[i : i + 1, name_len:, :]
        #     ctx_i = ctx[i : i + 1, :, :]
        #     prompt = torch.cat(
        #         [
        #             prefix_i,  # (1, 1, dim)
        #             class_i,   # (1, name_len, dim)
        #             ctx_i,     # (1, n_ctx, dim)
        #             suffix_i,  # (1, *, dim)
        #         ],
        #         dim=1,
        #     )
        #     prompts.append(prompt)
        # prompts = torch.cat(prompts, dim=0)

        return prompts
    
     
    def forward(self, im_features, data):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx   

        # print(f'im_features : {data}')
        # exit()

        # style = self.rblock(data) 
        # print(f'style : {len(style)}')
        # style = torch.Tensor(style) 
        # print(f'style : {len(style)}')
        # exit()
        # features = torch.cat((style, im_features),1)           
        
        #style = self.nnet(data)
        # style = self.style(data)
        #style = style.unsqueeze(1)

        multi = self.multi(data)
        #im_features = im_features.unsqueeze(1)
        # print(f'style : {style.shape}')
        # print(f'im_features : {im_features.shape}')
        # exit()
        # fcs = torch.cat((style,im_features),1) 

        # final_features = self.projector(im_features)
        # im_features = im_features.to(self.device)

        # bn = torch.nn.BatchNorm1d(512).cuda()
        # final_features = bn(im_features)

        # # print(f'final_features : {final_features.shape}')
        # # exit()

        # final_features = final_features.unsqueeze(1)

        im_features = im_features.unsqueeze(1)

        final_features = self.projector(im_features)

        fcs = torch.cat((multi,final_features),1) 
        #print(f'fcs : {fcs.shape}')
        # fcs = fcs.view(4, 1, -1)
        # fcs = fcs.view(4, 1, 6566)
        # exit()
        bias = self.injection(fcs) 
        # print(f'shape of bias : {bias.shape}')
        # print(f'bias : {bias}')
        bias = bias.to(torch.float16)
        # bias1 = self.linearlayer(bias.type(torch.float))
        # print(f'shape of bias1 : {bias1.shape}')
        # exit()
        # print(bias.dtype) 
        #print(f'shape of ctx : {ctx.shape}')    
        alpha1 = self.meta_net(bias.type(torch.float))
        alpha2 = self.meta_net(bias.type(torch.float))
        alpha3 = self.meta_net(bias.type(torch.float))
        alpha4 = self.meta_net(bias.type(torch.float))
        #print(f'shape of alpha1 : {alpha1.shape}')

        alpha_a = torch.cat((alpha1, alpha2),1)
        alpha_b = torch.cat((alpha_a, alpha3),1)
        alpha = torch.cat((alpha_b, alpha4),1)
        #print(f'shape of alpha : {alpha.shape}')

        ctx = ctx.unsqueeze(0)   
        #print(f'shape of ctx : {ctx.shape}') 
        #print(f'ctx : {ctx}')          
        # ctx_shifted = ctx + alpha 
        # print(f'shape of ctx_shifted : {ctx_shifted.shape}')  

        ctx_shifted = torch.add(ctx,alpha)
        # print(f'shape of ctx_shifted : {len(ctx_shifted)}')
        # exit()
        #print(f'ctx_shifted : {ctx_shifted}')       
               
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            #print('{ctx_shifted_i : {ctx_shifted_i.shape}')
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            #print(f'shape of ctx_i : {ctx_i.shape}') 
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)
            #print(f'shape of pts_i : {pts_i.shape}') 
            prompts.append(pts_i)
            # print(f'shape of prompts_i : {len(prompts)}') 
            # exit()
        prompts = torch.stack(prompts)

        # # print(f'shape of prompts : {prompts.shape}') 
        # transprompts = ctx_shifted.permute(0,2,1)
        # #print(f'shape of transprompts : {transprompts.shape}') 

        # pmul = torch.matmul(ctx_shifted, transprompts)
        # #print(f'shape of pmul : {pmul.shape}') 

        # identity = torch.eye(4)
        # mask = identity.repeat(4, 1, 1)
        # print(mask.shape)
        # print(mask)
        # exit()
        
        return prompts, ctx_shifted


class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label):
        global df 
        global cls_label

        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features, data = self.image_encoder(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts, ctx_shifted = self.prompt_learner(image_features, data)
  
        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        # t = image_features.squeeze(1).cpu().detach().numpy()
        t = logits.cpu().detach().numpy()
        label_np = label.cpu().detach().numpy()
        df1 = pd.DataFrame(t)
        df2 = pd.DataFrame(label_np)
        # print('df1 shape:',df1.shape)
        # df1[''] = label
        # df = df.append(df1, ignore_index=True)   
        # cls_label = cls_label.append(df2, ignore_index=True)    
        df = pd.concat([df, df1], ignore_index=True)
        cls_label = pd.concat([cls_label, df2], ignore_index=True) 
        df.to_csv('/home/dgxadmin/Ankit/ODG/csv/pacs/photo_img.csv', header=False, index=False) 
        cls_label.to_csv('/home/dgxadmin/Ankit/ODG/csv/pacs/photo_label.csv', header=False, index=False) 
        print('df :',df.shape) 
        print('cls_label :', cls_label.shape)

        return logits, ctx_shifted


class CustomCLIPtest(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label):
        # global df 
        # global cls_label

        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features, data = self.image_encoder(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # t = image_features.squeeze(1).cpu().detach().numpy()
        # t = image_features.cpu().detach().numpy()
        # label_np = label.cpu().detach().numpy()
        # df1 = pd.DataFrame(t)
        # df2 = pd.DataFrame(label_np)
        # # print('df1 shape:',df1.shape)
        # # df1[''] = label
        # # df = df.append(df1, ignore_index=True)   
        # # cls_label = cls_label.append(df2, ignore_index=True)    
        # df = pd.concat([df, df1], ignore_index=True)
        # cls_label = pd.concat([cls_label, df2], ignore_index=True) 
        # df.to_csv('/home/dgxadmin/Ankit/ODG/csv/pacs/sketch_img.csv', header=False, index=False) 
        # cls_label.to_csv('/home/dgxadmin/Ankit/ODG/csv/pacs/sketch_label.csv', header=False, index=False) 

        prompts, ctx_shifted = self.prompt_learner(image_features, data)
  
        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        return logits, ctx_shifted


# class AppleLoss(_Loss):
#     def __init__(self, T):
#         super(AppleLoss, self).__init__()
#         self.T = T

#     def forward(self, logits, ctx_shifted, label):
#         ce_loss = F.cross_entropy(logits, label)
        
#         transprompts = ctx_shifted.permute(0,2,1)
#         pmul = torch.matmul(ctx_shifted / self.T, transprompts / self.T)

#         identity = torch.eye(4).cuda()
#         mask = identity.repeat(4, 1, 1).cuda()
        
#         sim_loss = torch.linalg.det(torch.sub(pmul / self.T , mask / self.T)).mean()

#         total_loss = ce_loss + 0.05*torch.log10(-sim_loss)   # lambda = 0.1 : weight
#         # print(f'total_loss : {total_loss}')

#         return total_loss

