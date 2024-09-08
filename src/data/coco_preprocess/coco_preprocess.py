import os
import numpy as np
from PIL import Image
from tqdm import tqdm

image_folder = "/mnt/sda/minkyukim/coco2017/image_val"
mask_folder = "/mnt/sda/minkyukim/coco2017/label_val"

output_image_folder = "/mnt/sda/minkyukim/sam_dataset/coco_npy_test_dataset_1024image/imgs"
output_mask_folder = "/mnt/sda/minkyukim/sam_dataset/coco_npy_test_dataset_1024image/gts"

os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

def process_image(image_path, output_path):
    img = Image.open(image_path).convert('RGB')  # 알파 채널 제거하고 RGB로 변환
    img_resized = img.resize((1024, 1024))

    img_array = np.array(img_resized, dtype=np.float32) / 255.0

    np.save(output_path, img_array)

def process_mask(mask_path, output_path):
    mask = Image.open(mask_path).convert('L')  # 단일 채널로 변환 (흑백 이미지)
    mask_resized = mask.resize((256, 256))

    mask_array = np.array(mask_resized, dtype=np.float32) / 255.0

    np.save(output_path, mask_array)

image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
for image_file in tqdm(image_files, desc="Processing Images"):
    number = image_file.split('_')[1].split('.')[0]
    input_image_path = os.path.join(image_folder, image_file)
    output_image_path = os.path.join(output_image_folder, f"{number}.npy")
    
    process_image(input_image_path, output_image_path)

mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.png')]
for mask_file in tqdm(mask_files, desc="Processing Masks"):
    number1, number2 = mask_file.split('_')[1], mask_file.split('_')[2].split('.')[0]
    input_mask_path = os.path.join(mask_folder, mask_file)
    output_mask_path = os.path.join(output_mask_folder, f"{number1}_{number2}.npy")
    
    process_mask(input_mask_path, output_mask_path)

print("Processing complete!")

