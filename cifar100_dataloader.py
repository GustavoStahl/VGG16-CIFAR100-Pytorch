import os
import pickle 
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

torch.manual_seed(123)
np.random.seed(123)

def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

class CIFAR100Dataset(Dataset):
    def __init__(self, dataset_path, is_train=True):
        split_name = "train" if is_train else "test"
        
        split_path = os.path.join(dataset_path, split_name)
        meta_path = os.path.join(dataset_path, "meta")
               
        split_data = unpickle(split_path)
        meta = unpickle(meta_path)[b"fine_label_names"]
        
        labels = np.array(split_data[b"fine_labels"])
        images = np.array(split_data[b"data"])
        
        # from flatten to 3D grid
        images = images.reshape(-1, 3, 32, 32)
               
        # RRR...GGG...BBB to RGBRGBRGB...
        images = images.transpose((0,2,3,1))
                
        std = images.std(axis=(0,1,2)) / 255.
        mean = images.mean(axis=(0,1,2)) / 255.
           
        transform_list = [transforms.ToPILImage("RGB"),
                          transforms.Resize((224, 224)), 
                          transforms.ToTensor(),
                          transforms.Normalize(mean, std)]
        
        if is_train:
            transform_list += [transforms.RandomHorizontalFlip(.5)]
            
        transform = transforms.Compose(transform_list)
        
        self.std = std
        self.mean = mean
        self.meta = meta
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __getitem__(self, index):
        image = self.images[index].copy()
        label = self.labels[index].copy()
        return self.transform(image), label
        
    def __len__(self):
        return len(self.images)
    
    def tensor_to_image(self, image_norm):
        image_norm_np = image_norm.detach().cpu().numpy()
        image_norm_transposed = image_norm_np.transpose(1,2,0) # CWH -> WHC
        image = ((image_norm_transposed * self.std) + self.mean) * 255.
        return np.clip(image,0,255).astype(np.uint8)