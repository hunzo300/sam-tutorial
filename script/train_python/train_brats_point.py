# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import autorootcwd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import scipy.ndimage
from scipy.ndimage import zoom

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


def show_mask(mask, ax, random_color=False):
    mask = mask.astype(np.float32)
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]

    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20, num_points=2):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.num_points = num_points  # Number of points to sample
        
        # Determine dataset type based on data_root subfolder name
        if 'brats' in data_root.lower():
            self.dataset_type = 'brats'
        elif 'coco' in data_root.lower():
            self.dataset_type = 'coco'
        elif 'ivdm' in data_root.lower():
            self.dataset_type = 'ivdm'
        else:
            raise ValueError("Unknown dataset type. The dataset should contain 'brats', 'coco', or 'ivdm' in the path.")
        
        # Load ground truth files and filter based on dataset type
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        )
        if self.dataset_type in ['brats', 'ivdm']:
            self.gt_path_files = [
                file
                for file in self.gt_path_files
                if os.path.isfile(join(self.img_path, '_'.join(os.path.basename(file).split('_')[:2]) + ".npy"))
            ]
        elif self.dataset_type == 'coco':
            self.gt_path_files = [
                file
                for file in self.gt_path_files
                if os.path.isfile(join(self.img_path, os.path.basename(file).split('_')[0] + ".npy"))
            ]

        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])

        # Adjust image name processing based on dataset type
        if self.dataset_type in ['brats', 'ivdm']:
            img_name = img_name.split('_')[0] + '_' + img_name.split('_')[1] + '.npy'
        elif self.dataset_type == 'coco':
            img_name = img_name.split('_')[0] + '.npy'

        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"

        gt = np.load(self.gt_path_files[index], "r", allow_pickle=True)  # (256, 256)
        gt_1024 = zoom(gt, (4, 4), order=0)  # Resize to (1024, 1024) with nearest-neighbor

        gt_1024 = (gt_1024 > 0).astype(np.uint8)

        label_ids = np.unique(gt_1024)[1:]  # Skip the background (assumed to be 0)
        gt2D = np.uint8(gt_1024 == random.choice(label_ids.tolist()))  # only one label, (1024, 1024)
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"

        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        
        # Get random points from the mask
        point_coords, point_labels = self.sample_points(gt2D, self.num_points, dilation_iterations=40)

        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            torch.tensor(point_coords).float(),
            torch.tensor(point_labels).long(),
            img_name,
        )

    def sample_points(self, mask, num_points=2, dilation_iterations=40):
        """
        GT 마스크 근처에서 배경 포인트를 샘플링하는 함수.
        
        Args:
            mask (np.ndarray): 2D 바이너리 마스크 (foreground: 1, background: 0)
            num_points (int): 총 샘플링할 포인트 수 (기본값은 4)
            dilation_iterations (int): 마스크를 확장하는 횟수 (확장 범위 설정)

        Returns:
            point_coords (np.ndarray): 샘플링된 포인트의 좌표, shape (num_points, 2)
            point_labels (np.ndarray): 샘플링된 포인트의 레이블 (1: foreground, 0: background)
        """
        # Create a binary structure for dilation
        structure = np.ones((3, 3), dtype=np.uint8)

        # Perform binary dilation to expand the mask
        dilated_mask = scipy.ndimage.binary_dilation(mask, structure=structure, iterations=dilation_iterations)

        # Define the background mask as the dilated mask excluding the original mask
        bg_mask = dilated_mask & (~mask)

        # Get foreground and background coordinates
        fg_coords = np.argwhere(mask == 1)[:, ::-1]  # (x, y) coordinates of foreground
        bg_coords = np.argwhere(bg_mask == 1)[:, ::-1]  # (x, y) coordinates of background

        # If background points overlap with foreground, remove those points
        bg_coords = np.array([coord for coord in bg_coords if mask[coord[1], coord[0]] == 0])

        # Determine the number of foreground and background points to sample
        num_fg = num_points // 2
        num_bg = num_points - num_fg

        # Randomly sample from foreground and background
        if len(fg_coords) > 0:
            fg_sampled = fg_coords[np.random.choice(len(fg_coords), size=num_fg, replace=True)]
        else:
            fg_sampled = np.empty((0, 2), dtype=int)  # Empty if no foreground
        
        if len(bg_coords) > 0:
            bg_sampled = bg_coords[np.random.choice(len(bg_coords), size=num_bg, replace=True)]
        else:
            bg_sampled = np.empty((0, 2), dtype=int)  # Empty if no background

        # Combine the sampled points and their corresponding labels
        point_coords = np.vstack((fg_sampled, bg_sampled))
        point_labels = np.hstack((np.ones(len(fg_sampled)), np.zeros(len(bg_sampled))))  # 1 for fg, 0 for bg

        # Shuffle the points and labels to mix foreground and background points
        indices = np.random.permutation(len(point_coords))
        point_coords = point_coords[indices]
        point_labels = point_labels[indices]

        return point_coords, point_labels




# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,
    default="/mnt/sda/minkyukim/sam_dataset_refined/brats_npy_train_dataset_1024image",
    help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "-checkpoint", type=str, default="work_dir/SAM/sam_vit_b_01ec64.pth"
)
# parser.add_argument('-device', type=str, default='cuda:0')
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="load pretrain model"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=1000)
parser.add_argument("-batch_size", type=int, default=5)
parser.add_argument("-num_workers", type=int, default=0)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument(
    "-use_wandb", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
parser.add_argument("--device", type=str, default="cuda:1")
args = parser.parse_args()

if args.use_wandb:
    import wandb

    wandb.login()
    wandb.init(
        project=args.task_name,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.tr_npy_path,
            "model_type": args.model_type,
        },
    )

# %% set up model for training
# device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = "../../../../../../mnt/sda/minkyukim/pth/sam-tutorial_brats"
#join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)


# %% sanity test of dataset class
tr_dataset = NpyDataset(args.tr_npy_path)
tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)

for step, (image, gt, bboxes, point_coords, point_labels, names_temp) in enumerate(tr_dataloader):
    print(image.shape, gt.shape, bboxes.shape, point_coords.shape, point_labels.shape)
    
    # show the example
    _, axs = plt.subplots(1, 2, figsize=(25, 25))
    
    # Randomly select an index to display
    idx = random.randint(0, 7)
    
    # Show image with mask and bbox
    axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(gt[idx].cpu().numpy(), axs[0])
    show_box(bboxes[idx].numpy(), axs[0])

    # Show foreground and background points
    points_fg = point_coords[idx][point_labels[idx] == 1].cpu().numpy()
    points_bg = point_coords[idx][point_labels[idx] == 0].cpu().numpy()
    
    # Plot foreground points in red and background points in blue
    axs[0].scatter(points_fg[:, 0], points_fg[:, 1], color='red', label='Foreground Points')
    axs[0].scatter(points_bg[:, 0], points_bg[:, 1], color='blue', label='Background Points')
    
    axs[0].axis("off")
    axs[0].set_title(names_temp[idx])
    
    # Second subplot (another random index)
    idx = random.randint(0, 7)
    axs[1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(gt[idx].cpu().numpy(), axs[1])
    show_box(bboxes[idx].numpy(), axs[1])

    # Plot foreground and background points for the second subplot
    points_fg = point_coords[idx][point_labels[idx] == 1].cpu().numpy()
    points_bg = point_coords[idx][point_labels[idx] == 0].cpu().numpy()

    axs[1].scatter(points_fg[:, 0], points_fg[:, 1], color='red', label='Foreground Points')
    axs[1].scatter(points_bg[:, 0], points_bg[:, 1], color='blue', label='Background Points')

    axs[1].axis("off")
    axs[1].set_title(names_temp[idx])
    
    # Adjust and save the figure
    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.savefig("./data_sanitycheck_brats_point.png", bbox_inches="tight", dpi=300)
    plt.close()
    
    break



# %% set up model

class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box=None, point=None):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        
        # Prepare box and point inputs if provided
        if box is not None:
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)
        else:
            box_torch = None
        
        if point is not None:
            point_coords, point_labels = point
            point_coords = torch.as_tensor(point_coords, dtype=torch.float32, device=image.device)
            point_labels = torch.as_tensor(point_labels, dtype=torch.int64, device=image.device)
            points = (point_coords, point_labels)
        else:
            points = None
        
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=box_torch,
                masks=None,
            )
        
        # Decode mask using image embedding and prompt encoder output
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        
        # Resize the low resolution masks to the original image size
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        
        return ori_res_masks



def main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    medsam_model.train()

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in medsam_model.parameters()),
    )  # 93735472
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
    )  # 93729252

    img_mask_encdec_params = list(medsam_model.image_encoder.parameters()) + list(
        medsam_model.mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )  # 93729252
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # %% train
    num_epochs = args.num_epochs
    iter_num = 0
    losses = []
    best_loss = 1e10
    train_dataset = NpyDataset(args.tr_npy_path)

    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        for step, (image, gt2D, boxes, point_coords, point_labels, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()  # Convert boxes to numpy
            points_np = point_coords.detach().cpu().numpy()  # Convert points to numpy
            point_labels_np = point_labels.detach().cpu().numpy()  # Convert point labels to numpy

            image, gt2D = image.to(device), gt2D.to(device)
            point_coords, point_labels = point_coords.to(device), point_labels.to(device)

            if args.use_amp:
                ## AMP (Automatic Mixed Precision)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    # Forward pass with both boxes and points
                    medsam_pred = medsam_model(image, boxes_np, (points_np, point_labels_np))
                    
                    # Calculate losses
                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())

                # Backward pass and optimization with AMP
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                # Forward pass with both boxes and points
                medsam_pred = medsam_model(image, boxes_np, (points_np, point_labels_np))
                
                # Calculate losses
                loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                
                # Backward pass and optimization without AMP
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            iter_num += 1

        epoch_loss /= step
        losses.append(epoch_loss)

    # Log or print losses here if necessary

        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss})
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        ## save the latest model
        # checkpoint = {
        #     "model": medsam_model.state_dict(),
        #     "optimizer": optimizer.state_dict(),
        #     "epoch": epoch,
        # }
        torch.save(sam_model.state_dict(), join(model_save_path, f"medsam_model_{epoch}_refined_point.pth"))
        ## save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # checkpoint = {
            #     "model": medsam_model.state_dict(),
            #     "optimizer": optimizer.state_dict(),
            #     "epoch": epoch,
            # }
            torch.save(sam_model.state_dict(), join(model_save_path, "medsam_model_best_refined_point.pth"))

        # %% plot loss
        plt.plot(losses)
        plt.title("Dice + Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(model_save_path, args.task_name + "train_loss.png"))
        plt.close()


if __name__ == "__main__":
    main()