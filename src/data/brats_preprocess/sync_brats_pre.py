import os
import numpy as np
import shutil
from tqdm import tqdm

def synchronize_imgs_with_gts(imgs_dir, gts_dir):
    imgs_output_dir = os.path.join(imgs_dir, '../synced_imgs')
    os.makedirs(imgs_output_dir, exist_ok=True)
    
    gts_files = sorted(os.listdir(gts_dir))
    
    for gts_file in tqdm(gts_files):
        # Parse the gt file name
        parts = gts_file.split('_')
        first_number = parts[0]
        img_class = parts[1]
        lbl_number = parts[2].replace('lbl', '').replace('.npy', '')
        
        # Construct corresponding image filename
        original_img_filename = f"{first_number}_{img_class}.npy"
        new_img_filename = f"{first_number}_{img_class}_{lbl_number}.npy"
        
        original_img_path = os.path.join(imgs_dir, original_img_filename)
        new_img_path = os.path.join(imgs_output_dir, new_img_filename)
        
        if not os.path.exists(original_img_path):
            print(f"Image file {original_img_path} does not exist. Skipping.")
            continue
        
        # Copy and rename the image file
        shutil.copyfile(original_img_path, new_img_path)

# Paths to the train dataset directories
train_imgs_dir = '/mnt/sda/minkyukim/sam_dataset/brats_npy_train_dataset/imgs'
train_gts_dir = '/mnt/sda/minkyukim/sam_dataset/brats_npy_train_dataset/gts'

# Paths to the test dataset directories
test_imgs_dir = '/mnt/sda/minkyukim/sam_dataset/brats_npy_test_dataset/imgs'
test_gts_dir = '/mnt/sda/minkyukim/sam_dataset/brats_npy_test_dataset/gts'

# Synchronize the train dataset
print("Synchronizing train dataset images...")
synchronize_imgs_with_gts(train_imgs_dir, train_gts_dir)

# Synchronize the test dataset
print("Synchronizing test dataset images...")
synchronize_imgs_with_gts(test_imgs_dir, test_gts_dir)

print("Image synchronization complete.")
