import os
import numpy as np
from PIL import Image
from tqdm import tqdm

image_folder = "/mnt/sda/minkyukim/IVDM3Seg/preprocessed_image_test"
mask_folder = "/mnt/sda/minkyukim/IVDM3Seg/preprocessed_label_test"

output_image_folder = "/mnt/sda/minkyukim/sam_dataset/ivdm_npy_test_dataset_1024image/imgs"
output_mask_folder = "/mnt/sda/minkyukim/sam_dataset/ivdm_npy_test_dataset_1024image/gts"


os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

def process_image(image_path, output_path):
    img = Image.open(image_path).convert('RGB')  # 알파 채널 제거하고 RGB로 변환
    img_resized = img.resize((1024, 1024))

    img_array = np.array(img_resized, dtype=np.float32) / 255.0

    np.save(output_path, img_array)

def process_mask(mask_path):
    mask = Image.open(mask_path).convert('L')  # 단일 채널로 변환 (흑백 이미지)
    mask_resized = mask.resize((256, 256))

    mask_array = np.array(mask_resized, dtype=np.float32) / 255.0

    return mask_array


for class_folder in os.listdir(image_folder):
    class_path = os.path.join(image_folder, class_folder)
    
    if os.path.isdir(class_path):
        number1, class_name = class_folder.split('_')  # 폴더명에서 number1과 class 추출

        image_files = [f for f in os.listdir(class_path) if f.endswith('.png')]
        for image_file in tqdm(image_files, desc=f"Processing {class_folder}"):
            number2 = image_file.split('_')[1].split('.')[0]  # slice_{number}에서 number 추출
            number2 = str(int(number2)+1)
            input_image_path = os.path.join(class_path, image_file)
            
            # 저장할 파일명: {number1}-{number2}_{class}.npy
            output_image_path = os.path.join(output_image_folder, f"{number1}-{number2}_{class_name}.npy")
            
            # 이미지 처리 및 저장
            process_image(input_image_path, output_image_path)

classes = ["fat", "inn", "opp", "wat"]

mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.png')]

for mask_file in tqdm(mask_files, desc="Processing Masks"):
    # 파일명에서 {number1}_{number2}_{number3}.png 추출
    number1, number2, number3 = mask_file.split('_')
    number3 = number3.split('.')[0]  # 확장자 제거
    
    input_mask_path = os.path.join(mask_folder, mask_file)
    
    mask_array = process_mask(input_mask_path)
    
    for class_name in classes:
        output_mask_path = os.path.join(output_mask_folder, f"{number1}-{number2}_{class_name}_{number3}.npy")
        np.save(output_mask_path, mask_array)


print("Processing complete!")