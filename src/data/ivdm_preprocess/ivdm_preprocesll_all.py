import numpy as np
import os
from skimage.transform import resize

input_img_folder = "/mnt/sda/minkyukim/IVDM3Seg/train_seperated/imgs"
output_img_folder = "/mnt/sda/minkyukim/sam_dataset_refined/ivdm_npy_train_dataset_256image/imgs"
input_gt_folder = "/mnt/sda/minkyukim/IVDM3Seg/train_seperated/gts"
output_gt_folder = "/mnt/sda/minkyukim/sam_dataset_refined/ivdm_npy_train_dataset_256image/gts"

os.makedirs(output_img_folder, exist_ok=True)
os.makedirs(output_gt_folder, exist_ok=True)

classes = ['fat', 'inn', 'opp', 'wat']

def resize_and_save_images():
    for filename in os.listdir(input_img_folder):
        if filename.endswith(".npy"):

            filepath = os.path.join(input_img_folder, filename)
            number_class = filename.split('.')[0]  
            
  
            img = np.load(filepath)
            
  
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
            
            resized_img = resize(img, (256, 256), anti_aliasing=True)
   
            resized_img_rgb = np.stack([resized_img]*3, axis=-1)
            

            output_path = os.path.join(output_img_folder, filename)
            np.save(output_path, resized_img_rgb)
            print(f"저장 완료: {output_path}")


def resize_and_copy_masks():
    for filename in os.listdir(input_gt_folder):
        if filename.endswith(".npy"):
           
            filepath = os.path.join(input_gt_folder, filename)
            number, class_name, mask_number = filename.split('.')[0].split('_')
            
    
            gt_mask = np.load(filepath)

            output_mask_path = os.path.join(output_gt_folder, f"{number}_{class_name}_{mask_number}.npy")
            np.save(output_mask_path, gt_mask)
            print(f"GT 마스크 저장 완료: {output_mask_path}")

# 1. 이미지 리사이즈 및 저장 실행
resize_and_save_images()

# 2. GT 마스크 리사이즈 및 복사 실행
resize_and_copy_masks()
