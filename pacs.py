import os.path as osp

import torch
import torch.nn.utils as utils
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms.functional as TF
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import os
import glob 
import random
from trainers.odgclip_new import *

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import torchvision.transforms as transforms

device = "cuda:0" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device='cpu')

repeat_transform = transforms.Compose([
    transforms.ToTensor(),
])

class Office_Train(Dataset):
  def __init__(self,train_image_paths,train_domain,train_labels):
    self.image_path=train_image_paths
    self.domain=train_domain
    self.labels=train_labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self,idx):
    orig_img = Image.open(self.image_path[idx])
    orig_img1 = repeat_transform(orig_img)
    image = preprocess(orig_img)
    domain=self.domain[idx] 
    domain=torch.from_numpy(np.array(domain)) 
    label=self.labels[idx] 
    label=torch.from_numpy(np.array(label)) 
 
    label_one_hot=F.one_hot(label,num_classes)
  
    return orig_img1, image, domain, label, label_one_hot 


#################-------DATASET------#######################

domains = ['art_painting', 'cartoon', 'photo', 'sketch']

'''
############### The source dataset 1 ##################
'''

image_path_dom1=[]
label_class_dom1=[]
label_dom1=[]
class_names1=[]
path_dom1='/home/1098363-z100/nfs/users/Singha/testodg/data/pacs/art_painting'
domain_name1 = path_dom1.split('/')[-1]
dirs_dom1=os.listdir(path_dom1)
class_names = dirs_dom1
num_classes = len(class_names)
class_names.sort()
dirs_dom1.sort()
c=0
index=0
index_dom1 = [3, 0, 1]
# source_images_per_class = 32
for i in dirs_dom1:
    if index in index_dom1:
        class_names1.append(i)
        impaths = path_dom1 + '/' + i
        paths = glob.glob(impaths+'/**.jpg')
        random.shuffle(paths)
        #selected_paths = paths[:source_images_per_class]
        image_path_dom1.extend(paths)
        label_class_dom1.extend([c for _ in range(len(paths))])
    c = c + 1
    index = index + 1
label_dom1=[0 for _ in range(len(image_path_dom1))] 


'''
############### The source dataset 2 ##################
'''

image_path_dom2=[]
label_class_dom2=[]
label_dom2=[]
class_names2=[]
path_dom2='/home/1098363-z100/nfs/users/Singha/testodg/data/pacs/cartoon'
domain_name2 = path_dom2.split('/')[-1]
dirs_dom2=os.listdir(path_dom2)
dirs_dom2.sort()
c=0
index=0
index_dom2 = [4, 0, 2]
for i in dirs_dom2:
  if index in index_dom2:
    class_names2.append(i)
    impaths=path_dom2+'/' +i
    paths=glob.glob(impaths+'*/**.jpg')
    random.shuffle(paths)
    #selected_paths = paths[:source_images_per_class]
    image_path_dom2.extend(paths)
    label_class_dom2.extend([c for _ in range(len(paths))])
  c=c+1
  index=index+1  
label_dom2=[1 for _ in range(len(image_path_dom2))]  


'''
############### The source dataset 3 ##################
'''

image_path_dom3=[]
label_class_dom3=[]
label_dom3=[]
class_names3=[]
path_dom3='/home/1098363-z100/nfs/users/Singha/testodg/data/pacs/photo'
domain_name3 = path_dom3.split('/')[-1]
dirs_dom3=os.listdir(path_dom3)
dirs_dom3.sort()
c=0
index=0
index_dom3 = [5, 1, 2]
for i in dirs_dom3:
  if index in index_dom3:
    class_names3.append(i)
    impaths=path_dom3+'/' +i
    paths=glob.glob(impaths+'*/**.jpg')
    random.shuffle(paths)
    #selected_paths = paths[:source_images_per_class]
    image_path_dom3.extend(paths)
    label_class_dom3.extend([c for _ in range(len(paths))])
  c=c+1
  index=index+1
label_dom3=[2 for _ in range(len(image_path_dom3))]  

# Known Classes
index_dom = list(set(index_dom1 + index_dom2 + index_dom3))
known_class_names = [class_names[idx] for idx in index_dom]
known_classes = ",".join(known_class_names)
'''
############### The combining the source dataset ##################
'''   
  
image_path_final=[]
image_path_final.extend(image_path_dom1)
image_path_final.extend(image_path_dom2)
image_path_final.extend(image_path_dom3)
label_class_final=[]
label_class_final.extend(label_class_dom1)
label_class_final.extend(label_class_dom2)
label_class_final.extend(label_class_dom3)
label_dom_final=[]
label_dom_final.extend(label_dom1)
label_dom_final.extend(label_dom2)
label_dom_final.extend(label_dom3)
domain_names=[]
domain_names.append(domain_name1)
domain_names.append(domain_name2)
domain_names.append(domain_name3)
print("domain_names",domain_names)

    
'''
##### Creating dataloader ######
'''
batchsize = 4
train_prev_ds=Office_Train(image_path_final,label_dom_final,label_class_final)
print(f'length of train_prev_ds: {len(train_prev_ds)}')
train_dl=DataLoader(train_prev_ds,batch_size=batchsize, num_workers=2, shuffle=True)
orig_imgprev, img_prev, domain_prev, label_prev, label_prev_one_hot = next(iter(train_dl))

domain_prev = domain_prev.to(device)

class_names.sort()
train_prev_classnames = class_names[:6]


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def domain_mse_loss(logits, batch_images, batch_labels, batch_domains):
    # Pass the batch of images through the CLIP model

    mse_loss = 0.0
    count = 0

    for i in range(len(batch_images)):
        for j in range(i + 1, len(batch_images)):
            # Check if the label values are the same but the domain values are different
            if batch_labels[i] == batch_labels[j] and batch_domains[i] != batch_domains[j]:
                mse_loss += F.mse_loss(logits[i], logits[j])
                count += 1

    # Compute the average MSE loss
    if count > 0:
        mse_loss /= count

    return mse_loss
   
def hinge_loss(img1, img2, low, high):
  mse = torch.nn.MSELoss().to(device)
  output = mse(img1, img2)
  if output <= low:
    return low - output
  elif output > high:
    return output
  else:
    return torch.tensor(0.0)
  
def contrastive_loss_maximize_similar(feature1, feature2, margin=0.2):
    """
    Compute the contrastive loss to maximize the distance between similar pairs.

    Args:
        feature1 (torch.Tensor): A tensor of shape (batch_size, feature_dim) containing the features of the first set.
        feature2 (torch.Tensor): A tensor of shape (batch_size, feature_dim) containing the features of the second set.
        margin (float, optional): The margin parameter. Default is 0.2.

    Returns:
        torch.Tensor: The computed contrastive loss.
    """
    # Normalize the feature vectors
    feature1_normalized = F.normalize(feature1, p=2, dim=1)
    feature2_normalized = F.normalize(feature2, p=2, dim=1)
    
    # Compute cosine similarity
    similarity = F.cosine_similarity(feature1_normalized, feature2_normalized)
    
    # Compute contrastive loss
    loss = torch.mean(torch.clamp(margin - similarity, min=0.0))
    return loss

def domain_text_loss(diff_textfeatures, domain):
    losses = []
    for i in range(len(domain) - 1):
        if domain[i] != domain[i + 1]:
           loss = F.mse_loss(diff_textfeatures[i], diff_textfeatures[i + 1])
           losses.append(loss)

    mse_loss = torch.mean(torch.stack(losses))

    return mse_loss


class VisionEncoder(nn.Module):
    def __init__(self, clip1_model):
        super().__init__()
        self.image_encoder = clip1_model.visual
        self.dtype = clip1_model.dtype  

    def forward(self, image):
        im_features, data = self.image_encoder(image.type(self.dtype))
        im_features = im_features / im_features.norm(dim=-1, keepdim=True)
        return im_features, data

class ImageFilter(nn.Module):
    def __init__(self, brightness_threshold=0.01):
        super(ImageFilter, self).__init__()
        self.brightness_threshold = brightness_threshold

    def calculate_brightness(self, images):
        grayscale_images = torch.mean(images, dim=1, keepdim=True)  # Convert to grayscale
        return grayscale_images.mean((2, 3))  # Calculate the average pixel value for each image

    def forward(self, image_tensor):
        batch_size = image_tensor.size(0)
        brightness_values = self.calculate_brightness(image_tensor)

        fraction_to_select = 1.0
        
        num_images_to_select = int(batch_size * fraction_to_select)
        indices_with_brightness_condition = [i for i, value in enumerate(brightness_values) if value >= self.brightness_threshold]
        if len(indices_with_brightness_condition) < num_images_to_select:
           selected_indices = indices_with_brightness_condition
           num_black_images_to_select = num_images_to_select - len(indices_with_brightness_condition)
           all_indices = list(range(batch_size))
           black_indices = [i for i in all_indices if i not in indices_with_brightness_condition]
           random_black_indices = random.sample(black_indices, num_black_images_to_select)
           selected_indices += random_black_indices
           return selected_indices
        else:
           selected_indices = random.sample(indices_with_brightness_condition, num_images_to_select)
           return selected_indices

image_filter = ImageFilter(brightness_threshold=0.01)

def train_epoch(model, unknown_image_generator, classnames, domainnames, train_loader, optimizer, lr_scheduler, step):
    #param= sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(param)
    #param= sum(p.numel() for p in unknown_image_generator.parameters() if p.requires_grad)
    #print(param)
    #exit()
    loss_meter = AvgMeter()
    accuracy_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for orig_imgprev, img_prev, domain_prev, label_prev, label_one_hot_prev in tqdm_object:
        orig_imgprev = orig_imgprev.to(device)
        img_prev = img_prev.to(device)
        domain_prev = domain_prev.to(device)
        label_prev = label_prev.to(device)
        label_one_hot_prev = label_one_hot_prev.to(device)
        batch = img_prev.shape[0]

        unknown_posprompt1 = "art painting of an unknown object"
        stable_unknown_images1, generated_unknown_images1 = unknown_image_generator(batch, unknown_posprompt1, known_classes)

        unknown_posprompt2 = "cartoon of an unknown object"
        stable_unknown_images2, generated_unknown_images2 = unknown_image_generator(batch, unknown_posprompt2, known_classes)

        unknown_posprompt3 = "photo of an unknown object"
        stable_unknown_images3, generated_unknown_images3 = unknown_image_generator(batch, unknown_posprompt3, known_classes)

        # '''
        # Saved generated images of domain1
        # '''
        # generated_output_directory1 = '/home/1098363-z100/nfs/users/Singha/testodg/test_sketch/art_painting'
        # if not os.path.exists(generated_output_directory1):
        #    os.makedirs(generated_output_directory1)
        
        # generated_images_pil1 = [TF.to_pil_image(img) for img in stable_unknown_images1]
        # for i, img_pil in enumerate(generated_images_pil1):
        #    img_pil.save(os.path.join(generated_output_directory1, f"generated_image_dom1_{i}.jpg"))

        # '''
        # Saved generated images of domain2
        # '''
        # generated_output_directory2 = '/home/1098363-z100/nfs/users/Singha/testodg/test_sketch/cartoon'
        # if not os.path.exists(generated_output_directory2):
        #    os.makedirs(generated_output_directory2)
        
        # generated_images_pil2 = [TF.to_pil_image(img) for img in stable_unknown_images2]
        # for i, img_pil in enumerate(generated_images_pil2):
        #    img_pil.save(os.path.join(generated_output_directory2, f"generated_image_dom2_{i}.jpg"))

        # '''
        # Saved generated images of domain3
        # '''
        # generated_output_directory3 = '/home/1098363-z100/nfs/users/Singha/testodg/test_sketch/photo'
        # if not os.path.exists(generated_output_directory3):
        #    os.makedirs(generated_output_directory3)
        
        # generated_images_pil3 = [TF.to_pil_image(img) for img in stable_unknown_images3]
        # for i, img_pil in enumerate(generated_images_pil3):
        #    img_pil.save(os.path.join(generated_output_directory3, f"generated_image_dom3_{i}.jpg"))

        unknown_label_rank = len(train_prev_classnames)
        unknown_label = torch.full((len(domainnames)*generated_unknown_images1.shape[0],), unknown_label_rank).to(device)
        unknown_domain1 = torch.full((generated_unknown_images1.shape[0],), 0).to(device)
        unknown_domain2 = torch.full((generated_unknown_images2.shape[0],), 1).to(device)
        unknown_domain3 = torch.full((generated_unknown_images3.shape[0],), 2).to(device)


        # '''
        # Saved the batch images
        # '''
        # generated_output_directory4 = '/home/1098363-z100/nfs/users/Singha/testodg/test_sketch/original_images'
        # if not os.path.exists(generated_output_directory4):
        #    os.makedirs(generated_output_directory4)
        
        # generated_images_pil4 = [TF.to_pil_image(img) for img in orig_imgprev]
        # for i, img_pil in enumerate(generated_images_pil4):
        #    img_pil.save(os.path.join(generated_output_directory4, f"original_image_{i}.jpg"))


        generated_unknown_images = torch.cat((generated_unknown_images1, generated_unknown_images2, generated_unknown_images3), dim=0)
        unknown_domains = torch.cat((unknown_domain1, unknown_domain2, unknown_domain3), dim=0)
        random_indices = image_filter(generated_unknown_images) 
        selected_images = generated_unknown_images[random_indices]
        selected_labels = unknown_label[random_indices]
        selected_domains = unknown_domains[random_indices]
        
        img = torch.cat((img_prev, selected_images), dim=0)
        # img = torch.cat((img, generated_unknown_images2), dim=0)
        # img = torch.cat((img, generated_unknown_images3), dim=0)
        img = img.to(device)

        label = torch.cat((label_prev, selected_labels), dim=0)
        label = label.to(device)

        
        domain = torch.cat((domain_prev, selected_domains), dim=0)
        # domain = torch.cat((domain, unknown_domain2), dim=0)
        # domain = torch.cat((domain, unknown_domain3), dim=0)
        domain = domain.to(device)

        output, diff_projfeatures, latentimg = model(img, label)
        output = output.to(device)

        '''
         Saved original images
        '''
        generated_output_directory1 = '/home/1098363-z100/nfs/users/Singha/testodg/test_sketch2/original_images'
        if not os.path.exists(generated_output_directory1):
            os.makedirs(generated_output_directory1)
        
        generated_images_pil1 = [TF.to_pil_image(img) for img in orig_imgprev]
        for i, img_pil in enumerate(generated_images_pil1):
            img_pil.save(os.path.join(generated_output_directory1, f"original_image_{i}.jpg"))


        '''
        Saved preprocessed images
        '''
        generated_output_directory2 = '/home/1098363-z100/nfs/users/Singha/testodg/test_sketch2/clip_images'
        if not os.path.exists(generated_output_directory2):
            os.makedirs(generated_output_directory2)
        
        generated_images_pil2 = [TF.to_pil_image(img) for img in img]
        for i, img_pil in enumerate(generated_images_pil2):
            img_pil.save(os.path.join(generated_output_directory2, f"clip_image_{i}.jpg"))


        '''Saved latent images
        '''
        generated_output_directory5 = '/home/1098363-z100/nfs/users/Singha/testodg/test_sketch2/latent_images'
        if not os.path.exists(generated_output_directory5):
            os.makedirs(generated_output_directory5)
        
        generated_images_pil5 = [TF.to_pil_image(img) for img in latentimg]
        for i, img_pil in enumerate(generated_images_pil5):
            img_pil.save(os.path.join(generated_output_directory5, f"latent_image_{i}.jpg"))
    
        crossentropy_loss = F.cross_entropy(output, label)

        text_loss = domain_text_loss(diff_projfeatures, domain)

        loss = crossentropy_loss + (text_loss)
    
        optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(clip_model.parameters(), max_norm=1.0)
        optimizer.step()
        if step == "batch":
            lr_scheduler.step(loss_meter.avg)
        count = img.size(0)
        loss_meter.update(loss.item(), count)

        acc = compute_accuracy(output, label)[0].item()
        accuracy_meter.update(acc, count)
        print("accuracy:", accuracy_meter.avg)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, accuracy=accuracy_meter.avg, lr=get_lr(optimizer))
    return loss_meter, accuracy_meter.avg
  
unknown_image_generator = GenerateUnknownImages().to(device)
# vision_encoder = VisionEncoder(clip_model).to(device)

train_classnames = train_prev_classnames + ['unknown']
print(f'length of train_classnames : {len(train_classnames)}')

train_model = CustomCLIP(train_classnames, domain_names, clip_model).to(device)

params = [
            {"params": train_model.domainclass_pl.parameters()},
            {"params": train_model.domain_pl.parameters()},
            {"params": train_model.conv_layer.parameters()},
            {"params": train_model.upsample_net.parameters()}
        ]

optimizer = torch.optim.AdamW(params,  weight_decay=0.00001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=1, factor=0.8
        )
scaler = GradScaler() 

'''
Test dataset
'''
test_image_path_dom=[]
test_label_class_dom=[]
test_label_dom=[]
test_domain_names=[]
test_path_dom='/home/1098363-z100/nfs/users/Singha/testodg/data/pacs/sketch'
test_domain_name = test_path_dom.split('/')[-1]
test_dirs_dom=os.listdir(test_path_dom)
test_class_names = test_dirs_dom
test_num_classes = len(test_class_names)
test_dirs_dom.sort()
c=0
loc=0
index=0
text_index = [0,1,2,3,4,5,6]
for i in test_dirs_dom:
  if index in text_index:
    impaths=test_path_dom+'/' +i
    paths=glob.glob(impaths+'*/**.png')
    test_image_path_dom.extend(paths)
    test_label_class_dom.extend([c for _ in range(len(paths))])
  c=c+1
  index=index+1  
test_label_dom=[0 for _ in range(len(test_image_path_dom))]
  
test_image_path_final=[]
test_image_path_final.extend(test_image_path_dom)

test_label_class_final=[]
test_label_class_final_modified = [label if label <= 5 else 6 for label in test_label_class_dom]
test_label_class_final.extend(test_label_class_final_modified)

test_label_dom_final=[]
test_label_dom_final.extend(test_label_dom)


test_domain_names.append(test_domain_name)
test_domain_names.append(test_domain_name)
test_domain_names.append(test_domain_name)

'''
############### Making the test dataloader ##################
''' 

test_ds=Office_Train(test_image_path_final,test_label_dom_final,test_label_class_final)
print(len(test_ds))
test_dl=DataLoader(test_ds,batch_size=32, num_workers=4, shuffle=True)
_, test_img, test_domain, test_label, test_label_one_hot = next(iter(test_dl))

step = "epoch"
best_acc = 0
best_closed_set_acc = 0
best_open_set_acc = 0
best_avg_acc = 0
accuracy_file_path = "/home/1098363-z100/nfs/users/Singha/testodg/results2/pacs/sketch.txt"  
accuracy_dir = os.path.dirname(accuracy_file_path)
if not os.path.exists(accuracy_dir):
    os.makedirs(accuracy_dir)
accuracy_file = open(accuracy_file_path, "w")
torch.autograd.set_detect_anomaly(True)

for epoch in range(2):
    print(f"Epoch: {epoch + 1}")
    train_model.train()
    train_loss, train_acc = train_epoch(train_model, unknown_image_generator, train_classnames, domain_names, train_dl, optimizer, lr_scheduler, step)
    print(f"epoch {epoch+1} : training accuracy: {train_acc}")

    TRAIN_MODEL_PATH = Path("/home/1098363-z100/nfs/users/Singha/testodg/train_models2/pacs/sketch")
    TRAIN_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    TRAIN_MODEL_NAME = f"sketch_{epoch+1}.pth"
    TRAIN_MODEL_SAVE_PATH = TRAIN_MODEL_PATH / TRAIN_MODEL_NAME
    print(f"Saving train_model to: {TRAIN_MODEL_SAVE_PATH}")
    torch.save(obj=train_model.state_dict(), f=TRAIN_MODEL_SAVE_PATH)

    MODEL_PATH = "/home/1098363-z100/nfs/users/Singha/testodg/train_models2/pacs/sketch"
    MODEL_NAME = f"sketch_{epoch+1}.pth"
    MODEL_FILE = os.path.join(MODEL_PATH, MODEL_NAME)
    
    test_model = CustomCLIP(train_classnames, test_domain_names, clip_model).to(device)
    test_model.load_state_dict(torch.load(MODEL_FILE))

    with torch.no_grad():
        test_probs_all = torch.empty(0).to(device)
        test_labels_all = torch.empty(0).to(device)
        test_class_all = torch.empty(0).to(device)
        test_tqdm_object = tqdm(test_dl, total=len(test_dl))

        total_correct_a = 0
        total_samples_a = 0
        total_correct_b = 0
        total_samples_b = 0
        
        for orig_testimg, test_img, test_domain, test_label, test_label_one_hot in test_tqdm_object:
            test_img = test_img.to(device)
            orig_testimg = orig_testimg.to(device)
            test_domain =test_domain.to(device)
            test_label = test_label.to(device)
            test_label_one_hot = test_label_one_hot.to(device)
            
            test_output, _, latent_img = test_model(test_img.to(device), test_label)

            # '''
            # Saved original images
            # '''
            # generated_output_directory1 = '/home/1098363-z100/nfs/users/Singha/testodg/test_sketch/original_images'
            # if not os.path.exists(generated_output_directory1):
            #    os.makedirs(generated_output_directory1)
        
            # generated_images_pil1 = [TF.to_pil_image(img) for img in orig_testimg]
            # for i, img_pil in enumerate(generated_images_pil1):
            #    img_pil.save(os.path.join(generated_output_directory1, f"original_image_{i}.jpg"))


            # '''
            # Saved preprocessed images
            # '''
            # generated_output_directory2 = '/home/1098363-z100/nfs/users/Singha/testodg/test_sketch/clip_images'
            # if not os.path.exists(generated_output_directory2):
            #    os.makedirs(generated_output_directory2)
        
            # generated_images_pil2 = [TF.to_pil_image(img) for img in test_img]
            # for i, img_pil in enumerate(generated_images_pil2):
            #    img_pil.save(os.path.join(generated_output_directory2, f"clip_image_{i}.jpg"))


            # '''
            # Saved latent images
            # '''
            # generated_output_directory5 = '/home/1098363-z100/nfs/users/Singha/testodg/test_sketch/latent_images'
            # if not os.path.exists(generated_output_directory5):
            #    os.makedirs(generated_output_directory5)
        
            # generated_images_pil5 = [TF.to_pil_image(img) for img in latent_img]
            # for i, img_pil in enumerate(generated_images_pil5):
            #    img_pil.save(os.path.join(generated_output_directory5, f"latent_image_{i}.jpg"))

            predictions = torch.argmax(test_output, dim=1)
            class_a_mask = (test_label <= 5) 
            class_b_mask = (test_label > 5)

            correct_predictions_a = (predictions[class_a_mask] == test_label[class_a_mask]).sum().item()
            correct_predictions_b = (predictions[class_b_mask] == test_label[class_b_mask]).sum().item()
            
            total_correct_a += correct_predictions_a
            total_samples_a += class_a_mask.sum().item()
            
            total_correct_b += correct_predictions_b
            total_samples_b += class_b_mask.sum().item()
        
        closed_set_accuracy = total_correct_a / total_samples_a if total_samples_a > 0 else 0.0
        closed_set_acc = closed_set_accuracy*100
        open_set_accuracy = total_correct_b / total_samples_b if total_samples_b > 0 else 0.0
        open_set_acc = open_set_accuracy*100

        print(f"epoch {epoch+1} : open set prediction accuracy: {open_set_acc}")
        print(f"epoch {epoch+1} : closed set prediction accuracy: {closed_set_acc}")

        average_acc = (2*closed_set_acc*open_set_acc)/(closed_set_acc + open_set_acc)
        print(f"epoch {epoch+1} : average prediction accuracy: {average_acc}")

        accuracy_file.write(f"Epoch {epoch+1} - Open Set Accuracy: {open_set_acc}%\n")
        accuracy_file.write(f"Epoch {epoch+1} - Closed Set Accuracy: {closed_set_acc}%\n")
        accuracy_file.write(f"Epoch {epoch+1} - Average Accuracy: {average_acc}%\n")
        accuracy_file.write("\n") 
        accuracy_file.flush()
        
        if average_acc > best_avg_acc:
            best_closed_set_acc = closed_set_acc
            best_open_set_acc = open_set_acc
            best_avg_acc = average_acc
            TEST_MODEL_PATH = Path("/home/1098363-z100/nfs/users/Singha/testodg/test_models2/pacs")
            TEST_MODEL_PATH.mkdir(parents=True, exist_ok=True)
            TEST_MODEL_NAME = "sketch.pth"
            TEST_MODEL_SAVE_PATH = TEST_MODEL_PATH / TEST_MODEL_NAME
            print(f"Saving test_model with best average accuracy to: {TEST_MODEL_SAVE_PATH}")
            torch.save(obj=test_model.state_dict(), f=TEST_MODEL_SAVE_PATH) 
            
            print(f"Best open set prediction accuracy till now: {best_open_set_acc}")
            print(f"Best closed set prediction accuracy till now: {best_closed_set_acc}")
            print(f"Best average set prediction accuracy till now: {best_avg_acc}")
            accuracy_file.write(f"Epoch {epoch+1} - Best Open Set Accuracy till now : {best_open_set_acc}%\n")
            accuracy_file.write(f"Epoch {epoch+1} - Best Closed Set Accuracy till now: {best_closed_set_acc}%\n")
            accuracy_file.write(f"Epoch {epoch+1} - Best Average Accuracy now: {best_avg_acc}%\n")
            accuracy_file.write("\n") 
            accuracy_file.flush()
        else:
            print(f"Best open set prediction accuracy till now: {best_open_set_acc}")
            print(f"Best closed set prediction accuracy till now: {best_closed_set_acc}")
            print(f"Best average set prediction accuracy till now: {best_avg_acc}")
            accuracy_file.write(f"Epoch {epoch+1} - Best Open Set Accuracy till now : {best_open_set_acc}%\n")
            accuracy_file.write(f"Epoch {epoch+1} - Best Closed Set Accuracy till now: {best_closed_set_acc}%\n")
            accuracy_file.write(f"Epoch {epoch+1} - Best Average Accuracy now: {best_avg_acc}%\n")
            accuracy_file.write("\n") 
            accuracy_file.flush()

print(f"Best open set prediction accuracy till now: {best_open_set_acc}")
print(f"Best closed set prediction accuracy till now: {best_closed_set_acc}")
print(f"Best average set prediction accuracy till now: {best_avg_acc}")
