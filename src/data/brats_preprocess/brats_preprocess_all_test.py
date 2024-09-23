import numpy as np
import os
from skimage.transform import resize

# 입력 및 출력 폴더 경로 설정
input_img_folder = "/mnt/sda/minkyukim/Brats2020/test_seperated/imgs"
output_img_folder = "/mnt/sda/minkyukim/sam_dataset_refined/brats_npy_test_dataset_1024image/imgs"
input_gt_folder = "/mnt/sda/minkyukim/Brats2020/test_seperated/gts"
output_gt_folder = "/mnt/sda/minkyukim/sam_dataset_refined/brats_npy_test_dataset_1024image/gts"

# 출력 폴더가 없으면 생성
os.makedirs(output_img_folder, exist_ok=True)
os.makedirs(output_gt_folder, exist_ok=True)

# 클래스 리스트 정의
classes = ['T1', 'T2', 'T1ce', 'Flair']

# 1. 이미지 리사이즈 및 저장 함수 (1024x1024x3)
def resize_and_save_images():
    for filename in os.listdir(input_img_folder):
        if filename.endswith(".npy"):
            # 파일 경로 및 정보 추출
            filepath = os.path.join(input_img_folder, filename)
            number_class = filename.split('.')[0]  # 예: 1_T1
            
            # npy 파일 로드
            img = np.load(filepath)
            
            # (200, 200)의 이미지를 (1024, 1024)로 리사이즈
            resized_img = resize(img, (1024, 1024), anti_aliasing=True)
            
            # (1024, 1024, 3) 형식으로 변환 (3채널로 확장)
            resized_img_rgb = np.stack([resized_img]*3, axis=-1)
            
            # 새로운 파일 경로 설정 및 저장
            output_path = os.path.join(output_img_folder, filename)
            np.save(output_path, resized_img_rgb)
            print(f"저장 완료: {output_path}")

# 2. GT 마스크 리사이즈 및 복사 (256x256)
def resize_and_copy_masks():
    for filename in os.listdir(input_gt_folder):
        if filename.endswith(".npy"):
            # 파일 경로 및 정보 추출
            filepath = os.path.join(input_gt_folder, filename)
            number_mask = filename.split('.')[0]  # 예: 1_1
            
            # npy 파일 로드
            gt_mask = np.load(filepath)
            
            # (200, 200)의 마스크를 (256, 256)으로 리사이즈
            resized_mask = resize(gt_mask, (256, 256), anti_aliasing=False, preserve_range=True).astype(np.uint8)
            
            # 4개의 클래스별로 마스크 복사 및 저장
            for class_name in classes:
                output_mask_path = os.path.join(output_gt_folder, f"{number_mask}_{class_name}_lbl{filename.split('_')[1]}")
                np.save(output_mask_path, resized_mask)
                print(f"GT 마스크 저장 완료: {output_mask_path}")

# 1. 이미지 리사이즈 및 저장 실행
resize_and_save_images()

# 2. GT 마스크 리사이즈 및 복사 실행
resize_and_copy_masks()
