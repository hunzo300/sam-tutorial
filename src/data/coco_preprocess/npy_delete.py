import os

folder_path = "/mnt/sda/minkyukim/sam_dataset/coco_npy_train_dataset/gts"

files_to_delete = [
    "100226_3.npy",
    "112614_3.npy",
    "210804_1.npy",
    "499198_12.npy",
    "438629_5.npy",
    "426400_6.npy",
    "550395_8.npy"
]

for file_name in files_to_delete:
    file_path = os.path.join(folder_path, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_name} 삭제 완료")
    else:
        print(f"{file_name} 파일을 찾을 수 없습니다")

print("모든 파일 삭제 완료")
