import numpy as np
import os
from skimage.transform import resize

input_img_folder = "/mnt/sda/minkyukim/Brats2020/test_seperated/imgs"
output_img_folder = "/mnt/sda/minkyukim/sam_dataset_refined/brats_npy_test_dataset_1024image/imgs"
input_gt_folder = "/mnt/sda/minkyukim/Brats2020/test_seperated/gts"
output_gt_folder = "/mnt/sda/minkyukim/sam_dataset_refined/brats_npy_test_dataset_1024image/gts"

os.makedirs(output_img_folder, exist_ok=True)
os.makedirs(output_gt_folder, exist_ok=True)

classes = ['T1', 'T2', 'T1ce', 'Flair']

def resize_and_save_images():
    for filename in os.listdir(input_img_folder):
        if filename.endswith(".npy"):

            filepath = os.path.join(input_img_folder, filename)
            number_class = filename.split('.')[0] 
            
            img = np.load(filepath)
            
            resized_img = resize(img, (1024, 1024), anti_aliasing=True)
            
            resized_img_rgb = np.stack([resized_img]*3, axis=-1)
            
            output_path = os.path.join(output_img_folder, filename)
            np.save(output_path, resized_img_rgb)
            print(f"저장 완료: {output_path}")

def resize_and_copy_masks():
    for filename in os.listdir(input_gt_folder):
        if filename.endswith(".npy"):

            filepath = os.path.join(input_gt_folder, filename)
            number, mask_number = filename.split('.')[0].split('_')
            
            gt_mask = np.load(filepath)
            
            resized_mask = resize(gt_mask, (256, 256), anti_aliasing=False, preserve_range=True).astype(np.uint8)
            
            for class_name in classes:
                output_mask_path = os.path.join(output_gt_folder, f"{number}_{class_name}_lbl{mask_number}")
                np.save(output_mask_path, resized_mask)
                print(f"GT 마스크 저장 완료: {output_mask_path}")

# 1. 이미지 리사이즈 및 저장 실행
# resize_and_save_images()

# 2. GT 마스크 리사이즈 및 복사 실행
resize_and_copy_masks()
