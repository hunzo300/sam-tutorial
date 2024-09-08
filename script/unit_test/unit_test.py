import autorootcwd
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import torch.nn.functional as F

from script.train_ivdm import NpyDataset, show_mask, show_box, MedSAM, sam_model_registry, device, args, join

def unit_test():
    # dataset
    dataset = NpyDataset("/mnt/sda/minkyukim/sam_dataset/ivdm_npy_train_dataset_1024image")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # load model
    sam_model = sam_model_registry[args.model_type](checkpoint="/mnt/sda/minkyukim/pth/sam-tutorial_ivdm/medsam_model_best_new.pth")
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    medsam_model.eval()

    # 샘플 데이터 가져오기
    for step, (image, gt, bboxes, names_temp) in enumerate(dataloader):

        image = image.to(device)
        gt = gt.to(device)

        # predict code
        with torch.no_grad():
            # bboxes는 1024x1024 
            pred_mask = medsam_model(image, bboxes.numpy()) 

        os.makedirs("gt_images", exist_ok=True)

        gt_resized = F.interpolate(gt.float(), size=(1024, 1024), mode='nearest').squeeze(1)
        gt_resized_np = gt_resized[0].cpu().numpy()
        gt_resized_np = np.clip(gt_resized_np, 0, 1)  

        plt.figure(figsize=(10, 10))
        plt.imshow(gt_resized_np, cmap="gray")
        plt.axis("off")
        plt.savefig(f"gt_images/gt_{names_temp[0]}.png", bbox_inches="tight", pad_inches=0)
        plt.close()

        _, axs = plt.subplots(1, 3, figsize=(15, 5))

        # 원본 image
        image_np = image[0].cpu().permute(1, 2, 0).numpy()
        image_np = np.clip(image_np, 0, 1) 
        axs[0].imshow(image_np)
        show_box(bboxes[0].cpu().numpy(), axs[0])
        axs[0].set_title("Original Image with BBox")

        # GT
        axs[1].imshow(image_np)
        show_mask(gt_resized_np, axs[1])
        axs[1].set_title("GT Mask")

        # Predicted Mask
        axs[2].imshow(image_np)
        pred_mask_resized = F.interpolate(pred_mask.float(), size=(1024, 1024), mode='nearest').squeeze(1)
        pred_mask_resized_np = pred_mask_resized[0].cpu().numpy()
        pred_mask_resized_np = np.clip(pred_mask_resized_np, 0, 1)  
        show_mask(pred_mask_resized_np, axs[2])
        axs[2].set_title("Predicted Mask")

        plt.savefig(f"output_images/unit_test_output_{names_temp[0]}_new.png", bbox_inches="tight", dpi=300)
        plt.close()

        break

if __name__ == "__main__":
    unit_test()
