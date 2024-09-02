import os
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm

def process_dataset(images_dir, masks_dir, output_dir):
    imgs_output_dir = os.path.join(output_dir, 'imgs')
    gts_output_dir = os.path.join(output_dir, 'gts')
    
    # Create output directories if they don't exist
    os.makedirs(imgs_output_dir, exist_ok=True)
    os.makedirs(gts_output_dir, exist_ok=True)
    
    mask_files = os.listdir(masks_dir)
    
    for mask_file in tqdm(mask_files):
        
        number_class_part = '_'.join(mask_file.split('_')[:2])
        image_filename = f"{number_class_part}_img.png"
        
        image_path = os.path.join(images_dir, image_filename)
        mask_path = os.path.join(masks_dir, mask_file)
        
        if not os.path.exists(image_path):
            print(f"Image file {image_path} does not exist. Skipping.")
            continue
        
        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        mask = np.array(Image.open(mask_path), dtype=np.uint8)
        
        image_output_path = os.path.join(imgs_output_dir, f"{number_class_part}.npy")
        mask_output_path = os.path.join(gts_output_dir, f"{mask_file.split('.')[0]}.npy")
        
        np.save(image_output_path, image)
        np.save(mask_output_path, mask)

# Directories for train dataset
train_images_dir = '/mnt/sda/minkyukim/sam_dataset/images'
train_masks_dir = '/mnt/sda/minkyukim/sam_dataset/masks'
train_output_dir = '/mnt/sda/minkyukim/sam_dataset/brats_npy_train_dataset'

# Directories for test dataset
test_images_dir = '/mnt/sda/minkyukim/sam_dataset/image'
test_masks_dir = '/mnt/sda/minkyukim/sam_dataset/mask'
test_output_dir = '/mnt/sda/minkyukim/sam_dataset/brats_npy_test_dataset'

# Process train dataset
# print("Processing train dataset...")
# process_dataset(train_images_dir, train_masks_dir, train_output_dir)

# Process test dataset
print("Processing test dataset...")
process_dataset(test_images_dir, test_masks_dir, test_output_dir)

print("Dataset processing complete.")

