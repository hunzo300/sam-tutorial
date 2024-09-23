import autorootcwd
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from script.train_python.train_brats import NpyDataset, show_mask, show_box, MedSAM, sam_model_registry, device, args, join

def inference_on_npy(data_root, npy_file=None, bbox_shift=20):
    
    dataset = NpyDataset(data_root, bbox_shift=bbox_shift)
    
    if npy_file:
    
        dataset.gt_path_files = [join(dataset.gt_path, npy_file)]
        print(f"Running inference on specific npy file: {npy_file}")
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 모델 로드
    sam_model = sam_model_registry[args.model_type](checkpoint="/mnt/sda/minkyukim/pth/sam-tutorial_brats/medsam_model_best_refined.pth")
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    medsam_model.eval()

    # 데이터 추론 및 시각화
    for step, (image, gt, bboxes, img_name) in enumerate(dataloader):

        image = image.to(device)
        gt = gt.to(device)

        # 예측 수행
        with torch.no_grad():
            pred_mask = medsam_model(image, bboxes.numpy()) 

        # Ground Truth 저장 폴더 생성
        os.makedirs("gt_images", exist_ok=True)

        # GT 리사이징
        gt_resized = F.interpolate(gt.float(), size=(1024, 1024), mode='nearest').squeeze(1)
        gt_resized_np = gt_resized[0].cpu().numpy()
        gt_resized_np = np.clip(gt_resized_np, 0, 1)  

        # GT 저장
        plt.figure(figsize=(10, 10))
        plt.imshow(gt_resized_np, cmap="gray")
        plt.axis("off")
        plt.savefig(f"gt_images/gt_{img_name[0]}.png", bbox_inches="tight", pad_inches=0)
        plt.close()

        # 결과 시각화
        _, axs = plt.subplots(1, 3, figsize=(15, 5))

        # 원본 이미지
        image_np = image[0].cpu().permute(1, 2, 0).numpy()
        image_np = np.clip(image_np, 0, 1) 
        axs[0].imshow(image_np)
        show_box(bboxes[0].cpu().numpy(), axs[0])
        axs[0].set_title("Original Image with BBox")

        # GT 시각화
        axs[1].imshow(image_np)
        show_mask(gt_resized_np, axs[1])
        axs[1].set_title("GT Mask")

        # 예측된 마스크 시각화
        axs[2].imshow(image_np)
        pred_mask_resized = F.interpolate(pred_mask.float(), size=(1024, 1024), mode='nearest').squeeze(1)
        pred_mask_resized_np = pred_mask_resized[0].cpu().numpy()
        pred_mask_resized_np = np.clip(pred_mask_resized_np, 0, 1)  
        show_mask(pred_mask_resized_np, axs[2])
        axs[2].set_title("Predicted Mask")

        # 결과 저장
        os.makedirs("output_images", exist_ok=True)
        plt.savefig(f"output/output_images_refined/inference_output_{img_name[0]}_worst.png", bbox_inches="tight", dpi=300)
        plt.close()

        print(f"Inference completed for: {img_name[0]}")
        break 

if __name__ == "__main__":
    data_root = "/mnt/sda/minkyukim/sam_dataset_refined/brats_npy_test_dataset_1024image"
    npy_file = "1274_T1ce_lbl1.npy"  
    
    inference_on_npy(data_root, npy_file=npy_file)
