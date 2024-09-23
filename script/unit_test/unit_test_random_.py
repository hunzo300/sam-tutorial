import autorootcwd
import os
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from script.train_ivdm import NpyDataset, show_mask, show_box, MedSAM, sam_model_registry, device, args

def unit_test(random_seed=10):

    random.seed(random_seed)
    torch.manual_seed(random_seed)

    dataset = NpyDataset("/mnt/sda/minkyukim/sam_dataset/ivdm_npy_test_dataset_1024image")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)


    sam_model_1 = sam_model_registry[args.model_type](checkpoint="/home/minkyukim/sam-tutorial/work_dir/SAM/sam_vit_b_01ec64.pth")
    medsam_model_1 = MedSAM(
        image_encoder=sam_model_1.image_encoder,
        mask_decoder=sam_model_1.mask_decoder,
        prompt_encoder=sam_model_1.prompt_encoder,
    ).to(device)
    medsam_model_1.eval()


    sam_model_2 = sam_model_registry[args.model_type](checkpoint="/mnt/sda/minkyukim/pth/sam-tutorial_ivdm/medsam_model_best_new.pth")
    medsam_model_2 = MedSAM(
        image_encoder=sam_model_2.image_encoder,
        mask_decoder=sam_model_2.mask_decoder,
        prompt_encoder=sam_model_2.prompt_encoder,
    ).to(device)
    medsam_model_2.eval()

    for step, (image, gt, bboxes, names_temp) in enumerate(dataloader):

        image = image.to(device)
        gt = gt.to(device)

        with torch.no_grad():
            pred_mask_1 = medsam_model_1(image, bboxes.numpy()) 

        with torch.no_grad():
            pred_mask_2 = medsam_model_2(image, bboxes.numpy()) 

        os.makedirs("gt_images", exist_ok=True)

        gt_resized = F.interpolate(gt.float(), size=(1024, 1024), mode='nearest').squeeze(1)
        gt_resized_np = gt_resized[0].cpu().numpy()
        gt_resized_np = np.clip(gt_resized_np, 0, 1)  

        plt.figure(figsize=(10, 10))
        plt.imshow(gt_resized_np, cmap="gray")
        plt.axis("off")
        plt.savefig(f"gt_images/gt_{names_temp[0]}.png", bbox_inches="tight", pad_inches=0)
        plt.close()

        _, axs = plt.subplots(1, 4, figsize=(20, 5))

        image_np = image[0].cpu().permute(1, 2, 0).numpy()
        image_np = np.clip(image_np, 0, 1) 
        axs[0].imshow(image_np)
        show_box(bboxes[0].cpu().numpy(), axs[0])
        axs[0].set_title("Original Image with BBox")

        # GT
        axs[1].imshow(image_np)
        show_mask(gt_resized_np, axs[1])
        axs[1].set_title("GT Mask")

        # Predicted Mask from first checkpoint
        axs[2].imshow(image_np)
        pred_mask_resized_1 = F.interpolate(pred_mask_1.float(), size=(1024, 1024), mode='nearest').squeeze(1)
        pred_mask_resized_np_1 = pred_mask_resized_1[0].cpu().numpy()
        pred_mask_resized_np_1 = np.clip(pred_mask_resized_np_1, 0, 1)  
        show_mask(pred_mask_resized_np_1, axs[2])
        axs[2].set_title("Predicted Mask (pre-trained)")

        # Predicted Mask from second checkpoint
        axs[3].imshow(image_np)
        pred_mask_resized_2 = F.interpolate(pred_mask_2.float(), size=(1024, 1024), mode='nearest').squeeze(1)
        pred_mask_resized_np_2 = pred_mask_resized_2[0].cpu().numpy()
        pred_mask_resized_np_2 = np.clip(pred_mask_resized_np_2, 0, 1)  
        show_mask(pred_mask_resized_np_2, axs[3])
        axs[3].set_title("Predicted Mask (finetuned)")

        plt.savefig(f"/home/minkyukim/sam-tutorial/output_images/ivdm_{names_temp[0]}_comparison.png", bbox_inches="tight", dpi=300)
        plt.close()

        break

if __name__ == "__main__":
    unit_test(random_seed=30)
