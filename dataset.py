import os
import random
import glob
import io
import numpy as np
from PIL import Image
import torch

class WaymoDataset(torch.utils.data.Dataset):
    def __init__(self, raw_root, compressed_root, patch_size):
        self.patch_size = patch_size

        # Find all tfrecord folders
        self.raw_folders = sorted(glob.glob(os.path.join(raw_root, "tfrecord_*", "val")))
        self.compressed_folders = sorted(glob.glob(os.path.join(compressed_root, "tfrecord_*", "val")))
        
        # Verify matching number of folders
        assert len(self.raw_folders) == len(self.compressed_folders), \
            f"Number of raw folders ({len(self.raw_folders)}) doesn't match compressed folders ({len(self.compressed_folders)})"
        
        # Initialize lists to store all image paths
        self.compressed_images = []
        self.raw_images = []
        
        # Collect all image paths from all folders
        for raw_folder, comp_folder in zip(self.raw_folders, self.compressed_folders):
            # Get images from this folder pair
            compressed_imgs = sorted(glob.glob(os.path.join(comp_folder, "*.jpeg")))
            raw_imgs = sorted(glob.glob(os.path.join(raw_folder, "*.png"))) 
            
            # Verify matching number of images in this folder pair
            assert len(compressed_imgs) == len(raw_imgs), \
                f"Mismatch in images count for folders {raw_folder} ({len(raw_imgs)}) and {comp_folder} ({len(compressed_imgs)})"
            
            self.compressed_images.extend(compressed_imgs)
            self.raw_images.extend(raw_imgs)


    def __getitem__(self, idx):
        # load images and labels
        img_path = self.compressed_images[idx]
        label_path = self.raw_images[idx]
        
        img = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")
        
        
        image = np.array(img).astype(np.float32)
        image = np.transpose(image, axes=[2, 0, 1])
        image /= 255.0

        target = np.array(label).astype(np.float32)
        target = np.transpose(target, axes=[2, 0, 1])
        target /= 255.0
        
        return image, target

    def __len__(self):
        return len(self.compressed_images)
    
class CommaDataset(torch.utils.data.Dataset):
    def __init__(self, root, patch_size):
        self.patch_size = patch_size
        self.root = root

        # Find all folders
        self.raw_folders = sorted(glob.glob(os.path.join(root, "*", "raw")))
        self.compressed_folders = sorted(glob.glob(os.path.join(root, "*", "h264")))

        # Verify matching number of folders
        assert len(self.raw_folders) == len(self.compressed_folders), \
            f"Number of raw folders {len(self.raw_folders)} doesn't match compressed folders {len(self.compressed_folders)}"

        
        # Initialize lists to store all image paths
        self.compressed_images = []
        self.raw_images = []

        # Collect all image paths from all folders
        for raw_folder, comp_folder in zip(self.raw_folders, self.compressed_folders):
            # Get images from this folder pair
            compressed_imgs = sorted(glob.glob(os.path.join(comp_folder, "*.png")))
            raw_imgs = sorted(glob.glob(os.path.join(raw_folder, "*.png"))) 
            
            # Verify matching number of images in this folder pair
            assert len(compressed_imgs) == len(raw_imgs), \
                f"Mismatch in images count for folders {raw_folder} ({len(raw_imgs)}) and {comp_folder} ({len(compressed_imgs)})"
            
            self.compressed_images.extend(compressed_imgs)
            self.raw_images.extend(raw_imgs)

    def __getitem__(self, idx):
        # load images and labels
        img_path = self.compressed_images[idx]
        label_path = self.raw_images[idx]
        
        img = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")
        
        
        image = np.array(img).astype(np.float32)
        image = np.transpose(image, axes=[2, 0, 1])
        image /= 255.0

        target = np.array(label).astype(np.float32)
        target = np.transpose(target, axes=[2, 0, 1])
        target /= 255.0
        
        return image, target
    
    def __len__(self):
        return len(self.compressed_images)

        
class BSDSDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, patch_size, jpeg_quality, use_augmentation=False, train=True):
        self.patch_size = patch_size
        self.jpeg_quality = jpeg_quality
        self.use_augmentation = use_augmentation
        self.train = train

        # Find all tfrecord folders
        self.image_files = sorted(glob.glob(images_dir + "/*.jpg"))

    def __getitem__(self, idx):

        label = Image.open(self.image_files[idx]).convert("RGB")

        if self.use_augmentation:
            # random rescale
            if random.random() < 0.5:
                scale = random.choice([0.9,0.8,0.7,0.6])
                label = label.resize((int(label.width*scale), int(label.height*scale)), Image.BICUBIC)

            # random rotate
            if random.random() < 0.5:
                label = label.rotate(random.choice([90,180,270]), expand=True)

        
        # randomly crop patch from training set
        if self.train:
            x = random.randint(0, label.width - self.patch_size)
            y = random.randint(0, label.height - self.patch_size)
            label = label.crop((x, y, x + self.patch_size, y + self.patch_size))
        
        
        # jpeg noise
        buffer = io.BytesIO()
        label.save(buffer, format="JPEG", quality=self.jpeg_quality)
        input = Image.open(buffer).convert("RGB")
        
        input = np.array(input).astype(np.float32)
        input = np.transpose(input, axes=[2, 0, 1])
        input /= 255.0

        target = np.array(label).astype(np.float32)
        target = np.transpose(target, axes=[2, 0, 1])
        target /= 255.0
        
        return input, target

    def __len__(self):
        return len(self.image_files)


    