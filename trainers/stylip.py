import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
import torchvision.transforms as transforms
from diffusers import StableDiffusionPipeline
import logging
from clip import clip
from torchvision.transforms import ToTensor
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

device = "cuda:3" if torch.cuda.is_available() else "cpu"


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

class TextEncoder(nn.Module):
		def __init__(self, clip_model):
				super().__init__()
				self.transformer = clip_model.transformer
				# self.transformer.requires_grad=False
				self.positional_embedding = clip_model.positional_embedding
				self.ln_final = clip_model.ln_final
				self.text_projection = clip_model.text_projection
				self.dtype = clip_model.dtype

		def forward(self, prompts, tokenized_prompts):
				x = prompts + self.positional_embedding.type(self.dtype)
				x = x.permute(1, 0, 2)  # NLD -> LND
				x,_ = self.transformer(x)
				x = x.permute(1, 0, 2)  # LND -> NLD
				x = self.ln_final(x).type(self.dtype)

				# x.shape = [batch_size, n_ctx, transformer.width]
				# take features from the eot embedding (eot_token is the highest number in each sequence)
				x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

				return x

class style_prompt(nn.Module):
	def __init__(self):
		super(style_prompt,self).__init__()
		self.linear = nn.ModuleList(nn.Linear(768*3,512) for _ in range (12))
		self.adain=AdaIN_trans()
		self.gap=nn.AdaptiveAvgPool2d((1,1))
	def forward(self,data):
		data_prompt = []
		for i in range(len(data)):
			# print(data[0].shape,data[0].permute(1,0,2).shape)
			x_mu=self.adain.mu(data[i])
			x_sigma=self.adain.sigma(data[i])
			# print(x_mu.shape, torch.mean(data[i],1).squeeze(1).shape)
			# exit()
			x=torch.cat((x_mu,x_sigma,torch.mean(data[i],1).squeeze(1)),1)
			# print('style',x.shape)
			x=self.linear[i](x)
			data_prompt.append(x)
		data_prompt=torch.stack(data_prompt,1)  
		# print(data_prompt.shape)
		# exit()
		return data_prompt

class nnet(nn.Module):
	def __init__(self):
		super(nnet,self).__init__()
		self.gap=nn.AdaptiveAvgPool2d((1,1))
		self.lin1=nn.Linear(768,256)
		self.lin2 = nn.Linear(256*12,512)
		self.adain= AdaIN_trans()
	def forward(self,data):
		x = []
		for i in range(len(data)):
			x_mu = self.adain.mu(data[i])
			x_mu = self.lin1(x_mu)
			x.append(x_mu)
		x= torch.cat(x,1)
		# print('n_net',x.shape)
		# exit()
		x= self.lin2(x)
		return x

class embed(nn.Module):
	def __init__(self):
		super(embed,self).__init__()
		self.lin1=nn.Linear(1024,512)

	def forward(self,x):
		x=self.lin1(x)
		return x  

class PromptLearner(nn.Module):
	def __init__(self, classnames, clip_model):
		super().__init__()
		n_cls = len(classnames)
		n_ctx = 4
		dtype = clip_model.dtype
		ctx_dim = clip_model.ln_final.weight.shape[0]
		vis_dim = clip_model.visual.output_dim
		clip_imsize = clip_model.visual.input_resolution

		prompt_prefix = " ".join(["X"] * n_ctx)

		print(f'Initial context: "{prompt_prefix}"')
		print(f"Number of context words (tokens): {n_ctx}")

				# self.ctx = nn.Parameter(ctx_vectors)

		self.meta_net = style_prompt()

		classnames = [name.replace("_", " ") for name in classnames]
		name_lens = [len(_tokenizer.encode(name)) for name in classnames]
		prompts = [prompt_prefix + " " + name + "." for name in classnames]

		tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
		#tokenized_prompts = tokenized_prompts.to(device2)
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
				# self.token_projection = nn.Parameter(torch.empty(138, 77))
		
	def construct_prompts(self, ctx, prefix, suffix):
				# dim0 is either batch_size (during training) or n_cls (during testing)
				# ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
				# prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
				# suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

		prompts = torch.cat(
				[
					prefix,  # (dim0, 1, dim)
					ctx,     # (dim0, n_ctx, dim)
					suffix,  # (dim0, *, dim)
				],
				dim=1,
			)

		return prompts

	def forward(self, data):
		prefix = self.token_prefix
		suffix = self.token_suffix
		ctx_shifted = self.meta_net(data)
		ctx_shifted = torch.mean(ctx_shifted,dim=1).unsqueeze(1).repeat_interleave(4, dim=1)
	
		prompts = []
		for ctx_shifted_i in ctx_shifted:
			ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
			pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
			prompts.append(pts_i)
		prompts = torch.stack(prompts)
				
		return prompts
		
def concat(im_features,text_features, num_classes):
	mat=torch.empty((im_features.size(0),num_classes,1024))
	for i in range(im_features.size(0)):
		imf=im_features[i,:]
		txt=text_features[i,:,:]
		mat[i,:,512:]=txt
		mat[i,:,0:512]=imf
		
	return mat 

class CustomCLIP(nn.Module):
	def __init__(self, classnames, clip_model):
		super().__init__()
		self.prompt_learner = PromptLearner(classnames, clip_model)
		self.tokenized_prompts = self.prompt_learner.tokenized_prompts
		self.image_encoder = clip_model.visual
		self.text_encoder = TextEncoder(clip_model)
		self.logit_scale = clip_model.logit_scale
		self.dtype = clip_model.dtype
		self.net_sty = nnet()
		self.lin_embed=embed()
		self.num_classes = len(classnames)
		
	def forward(self, image,label=None):
		tokenized_prompts = self.tokenized_prompts
		logit_scale = self.logit_scale.exp()
		image_features, data = self.image_encoder(image.type(self.dtype))
		image_features = image_features / image_features.norm(dim=-1, keepdim=True)
		prompts = self.prompt_learner(data)		
		f_ms=self.net_sty(data)
		
		text_features = []
		for pts_i in prompts:
			tf = self.text_encoder(pts_i, tokenized_prompts)
			text_features.append(tf)
		text_features=torch.stack(text_features)
		# print(text_features.shape, f_ms.shape)
		f_st=concat(f_ms,text_features, self.num_classes)
		f_st1=self.lin_embed(f_st.to(device))

		f_st1 = f_st1 / f_st1.norm(dim=1, keepdim=True)
		logits=[]
		# print(f_st.shape, features.shape)
		for txt,im in zip(f_st1,image_features):
			# print(im.shape, txt.shape, logit_scale.shape)
			l_i = logit_scale * im @ txt.t()
			logits.append(l_i)
		logits = torch.stack(logits) 
		return logits

class StableDiffusion(nn.Module):
    def __init__(self):
        super(StableDiffusion, self).__init__()
        self.model_id_or_path = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id_or_path, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(device)
        logging.basicConfig(level=logging.WARNING)
    def forward(self, images , pos_prompt, neg_prompt):       
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
        # self.unknown_prompt_learner = UnknownPL(clip_model) 
        # self.unknown_tokenized_prompts = self.unknown_prompt_learner.tokenized_prompts 
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

        resized_input_images = torch.stack([norm_transform(x) for x in image])
        #matched_texts = normalize(resized_input_images)
        matched_texts = resized_input_images.to(device)
        
        generated_unknown_images = self.diffusion(matched_texts.to(torch.float32), pos_prompt, neg_prompt) 
        resized_unknown_images = torch.stack([resize_transform(x) for x in generated_unknown_images])
        generated_unknown_images1 = normalize(resized_unknown_images)
        generated_unknown_images1 = generated_unknown_images1.to(device)

        return generated_unknown_images, generated_unknown_images1, matched_texts