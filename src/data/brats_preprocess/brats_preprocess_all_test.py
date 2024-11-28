# import numpy as np
# import os
# from skimage.transform import resize

# input_img_folder = "/mnt/sda/minkyukim/Brats2020/train_seperated/imgs"
# output_img_folder = "/mnt/sda/minkyukim/sam_dataset_refined/brats_npy_train_dataset_256image/imgs"
# input_gt_folder = "/mnt/sda/minkyukim/Brats2020/train_seperated/gts"
# output_gt_folder = "/mnt/sda/minkyukim/sam_dataset_refined/brats_npy_train_dataset_256image/gts"

# os.makedirs(output_img_folder, exist_ok=True)
# os.makedirs(output_gt_folder, exist_ok=True)

# classes = ['T1', 'T2', 'T1ce', 'Flair']

# def resize_and_save_images():
#     for filename in os.listdir(input_img_folder):
#         if filename.endswith(".npy"):

#             filepath = os.path.join(input_img_folder, filename)
#             number_class = filename.split('.')[0]  
            
#             img = np.load(filepath)
            
#             resized_img = resize(img, (256, 256), anti_aliasing=True)
            
#             # (1024, 1024, 3) 형식으로 변환 (3채널로 확장)
#             resized_img_rgb = np.stack([resized_img]*3, axis=-1)
            
#             output_path = os.path.join(output_img_folder, filename)
#             np.save(output_path, resized_img_rgb)
#             print(f"저장 완료: {output_path}")

# def resize_and_copy_masks():
#     for filename in os.listdir(input_gt_folder):
#         if filename.endswith(".npy"):

#             filepath = os.path.join(input_gt_folder, filename)
#             number_mask = filename.split('.')[0]  
            
#             gt_mask = np.load(filepath)
            
#             resized_mask = resize(gt_mask, (256, 256), anti_aliasing=False, preserve_range=True).astype(np.uint8)
            
#             for class_name in classes:
#                 output_mask_path = os.path.join(output_gt_folder, f"{number_mask}_{class_name}_lbl{filename.split('_')[1]}")
#                 np.save(output_mask_path, resized_mask)
#                 print(f"GT 마스크 저장 완료: {output_mask_path}")

# # 1. 이미지 리사이즈 및 저장 실행
# resize_and_save_images()

# # 2. GT 마스크 리사이즈 및 복사 실행
# resize_and_copy_masks()
import os
import glob

def rename_files_in_directory(directory):
    # .npy 파일 리스트 가져오기
    files = glob.glob(os.path.join(directory, "*.npy"))

    # 파일 이름 변경 루프
    for file_path in files:
        # 파일 이름 추출
        file_name = os.path.basename(file_path)
        
        # 파일 이름이 예상하는 패턴으로 되어 있는지 확인
        parts = file_name.split('_')
        if len(parts) >= 4:
            # number1, number2, classname, labelname 추출
            number1 = parts[0]
            classname = parts[2]
            labelname = parts[3]
            
            # 새 파일 이름 생성
            new_file_name = f"{number1}_{classname}_{labelname}"
            new_file_path = os.path.join(directory, new_file_name)
            
            # 파일 이름 변경
            os.rename(file_path, new_file_path)
            print(f"Renamed: {file_name} -> {new_file_name}")
        else:
            print(f"Skipping: {file_name}, does not match expected pattern")

# 디렉토리 경로
test_directory = "/mnt/sda/minkyukim/sam_dataset_refined/brats_npy_test_dataset_256image/gts"
train_directory = "/mnt/sda/minkyukim/sam_dataset_refined/brats_npy_train_dataset_256image/gts"

# 파일 이름 변경 실행
rename_files_in_directory(test_directory)
rename_files_in_directory(train_directory)

