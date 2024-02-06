import os
import gc
from tqdm import tqdm
import PIL
import clip
import torch

# import models
from models.model_grouping import GroupingVideoMAE
from models.patch_clip import PatchCLIP

# import data
from data import collate_func
import data.vid_transforms as TT
from data.vid_dataset_eval import VIDGroupingEvalDataset

# import other utils
from post_processing import visual_text_matching_ST
from utils.seeds import set_seed
from utils.mask2bbox import convert_to_boxlist, masks_to_boxes
from prompt import Prompts_with_synonym, template_sentence_list, COCO_stuff_class, VID_classes, VID_classes_dict
from evaluation.vid_eval import do_vid_evaluation_percent as do_vid_evaluation
from evaluation.grouping_metrics import compute_unsupervised_metrics
from visualization.vis_tools import get_seg_vis, get_bbox_vis, get_seg_vis_onehot, Denormalize
from datetime import datetime
import time

from utils.multi_process import synchronize, is_main_process
import torch.distributed as dist
from einops import rearrange
import torch.nn.functional as F
from skimage import color
import numpy as np
import argparse
from pprint import pprint
import copy
import json

def init_process_group():
    dist.init_process_group(
        backend='nccl',
        init_method='env://'),


def init_model(device, 
               is_distributed,
               num_slots,
               num_frames,
               st_grouping_init_ckpt_path,
               st_grouping_ckpt_path,
               clip_pacl_ckpt_path
               ):
    # init the grouping model
    print("Loading ST-Grouping")
    mae_grouping = GroupingVideoMAE(checkpoint_path=st_grouping_init_ckpt_path,
                                               object_dim=128,
                                               n_slots=num_slots,
                                               feat_dim=768,
                                               num_patches=196,
                                               num_frames=num_frames,
                                               img_size=224).to(device)
    print("Loading checkpoint from {}".format(st_grouping_ckpt_path))
    ckpt = torch.load(st_grouping_ckpt_path, map_location='cpu')
    # This line is designed to make the script compatible to the  square-based checkpoint.
    # It should have no effect on the multi-res trained checkpoint.
    ckpt["decoder.pos_embed"] = ckpt["decoder.pos_embed"].reshape(-1, 196, ckpt["decoder.pos_embed"].shape[-1])
    mae_grouping.load_state_dict(ckpt, strict=True)
    mae_grouping.eval()
    del ckpt
    gc.collect()

    # init vallina CLIP model
    print("Loading CLIP")
    clip_model, preprocess = clip.load("ViT-B/16", device='cpu')
    clip_model = clip_model.to(device)

    # init CLIP_PACL model
    print("Loading CLIP PACL")
    clip_pacl = PatchCLIP()
    print("Loading checkpoint from {}".format(clip_pacl_ckpt_path))
    ckpt = torch.load(clip_pacl_ckpt_path, map_location="cpu")['state_dict']
    new_dict = {}
    for k, v in ckpt.items():
        newk = k[7:]
        new_dict[newk] = v
    clip_pacl.load_state_dict(new_dict)
    clip_pacl = clip_pacl.to(device)
    clip_pacl.eval()
    del ckpt
    del new_dict
    gc.collect()

    return mae_grouping, clip_model, clip_pacl


def init_data(is_distributed, device, data_dir, dataset_img_dir, dataset_anno_dir, img_index_path, num_frames):
    print("Loading data")
    transforms = TT.Compose([
        TT.ToTensor(),
        TT.Resize2NearestPatch(scale=224, patch_size=16),
        TT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_bgr255=False),
    ])
    denorm = Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)
    val_dataset = VIDGroupingEvalDataset(img_dir=dataset_img_dir,
                                         anno_path=dataset_anno_dir,
                                         img_index=img_index_path,
                                         transforms=transforms,
                                         num_frames=num_frames,
                                         data_dir=data_dir,
                                         is_corp=False)

    if is_distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        val_sampler = torch.utils.data.SequentialSampler(val_dataset) 

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=val_sampler,
                                             num_workers=0,
                                             collate_fn=collate_func)
    return val_loader, val_dataset, denorm


def do_inference(val_loader, mae_grouping, clip_pacl, denorm,
                 prompts, foreground_prompts, vis, duplicate_box, device,
                 num_frames, n_stmae_seeds, n_spectral_clustering, refine_masks, filter_bg, merge_fg, n_batches, batch_offset):
    # do inference
    print("Start inference")
    results_dict, gt_dict = {}, {}
    with torch.no_grad():
        for batch_id, batch_dev in enumerate(tqdm(val_loader)):
            if batch_id < batch_offset:
                continue
            if n_batches > 0 and batch_id >= n_batches+batch_offset:
                break
            img_tsr, target_list, img_idx_list = batch_dev[0]       # [num_frames, C, H, W]
            Height, Width = img_tsr.shape[-2:]
            img_tsr = img_tsr.permute(0, 2, 3, 1).unsqueeze(0).to(device)        # [b, num_frames, H, W, C]

            # inference grouping
            masks = []
            masks_as_image = []
            for _ in range(n_stmae_seeds):
                mask, mask_as_image = mae_grouping.inference(img_tsr)    # [b, num_frames, K, num_patches], [b, num_frames, K, H, W]
                masks.append(mask)
                masks_as_image.append(mask_as_image)
            masks = torch.cat(masks)
            masks_as_image = torch.cat(masks_as_image)
            
            refine_masks = refine_masks if n_stmae_seeds > 1 else "false"
            foreground_mask_as_image, merged_mask_as_image, category_predictions, clustered_mask_as_image_one_hot = \
                visual_text_matching_ST(img_tsr[0], masks,
                                        prompts, foreground_prompts,
                                        clip_pacl,
                                        refine_masks, n_spectral_clustering,
                                        filter_bg, merge_fg,
                                        duplicate_box)
            for frm in range(num_frames):
                boxes = masks_to_boxes(merged_mask_as_image[frm])
                pred = convert_to_boxlist(boxes.cpu(), category_predictions, Height, Width, duplicate_box=duplicate_box)
                results_dict.update({img_idx_list[frm]: pred})
                gt_dict.update({img_idx_list[frm]: target_list[frm]})
            if vis:
                """
                Note that this is a quick debug option for ipynb
                Will display 20 images with [ground truth box, full mask, foreground mask, merged mask, pred box]
                """
                seg_img_pred = get_seg_vis(denorm(img_tsr[0]), masks_as_image[0]).cpu()  # [num_frames, C, H, W]
                
                cmask = clustered_mask_as_image_one_hot/clustered_mask_as_image_one_hot.max()
                seg_img_pred_clustered = get_seg_vis(denorm(img_tsr[0]), cmask).cpu()  # [num_frames, C, H, W]
                seg_img_pred_foreground = get_seg_vis_onehot(denorm(img_tsr[0]), foreground_mask_as_image).cpu()  # [num_frames, C, H, W]
                seg_img_pred_merge = get_seg_vis_onehot(denorm(img_tsr[0]), merged_mask_as_image).cpu()  # [num_frames, C, H, W]

                boxes = masks_to_boxes(merged_mask_as_image[0])
                boxes_on_image_pred = get_bbox_vis(denorm(img_tsr[0][0:1]), boxes.unsqueeze(0), 0, 1).cpu()
                boxes_on_image_gt = get_bbox_vis(denorm(img_tsr[0][0:1]), target_list[0].bbox.unsqueeze(0), 0, 1).cpu()

                PIL.Image.fromarray(
                            torch.cat([boxes_on_image_gt[0].permute(1, 2, 0),
                           seg_img_pred[0].permute(1, 2, 0), 
                           seg_img_pred_clustered[0].permute(1, 2, 0), 
                           seg_img_pred_foreground[0].permute(1, 2, 0), 
                           seg_img_pred_merge[0].permute(1, 2, 0),
                           boxes_on_image_pred[0].permute(1, 2, 0)], dim=0).numpy()).show()

                images_stmae_seeds = []
                for m in masks:
                    m = rearrange(m, 'a b (h w) -> a b h w', h=Height//16, w=Width//16)
                    m = F.interpolate(m, scale_factor=16, mode='nearest-exact')#[0,0,:,:]
                    m = m/m.max()
                    im = get_seg_vis(denorm(img_tsr[0]), m)[0].permute(1, 2, 0)
                    images_stmae_seeds.append(im)
                PIL.Image.fromarray(
                    torch.cat(images_stmae_seeds, dim=0).cpu().numpy()
                ).show()
        synchronize()
    return results_dict, gt_dict


def save_results(output_folder, results_dict, gt_dict, device_id):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    synchronize()
    
    print("Save results")
    torch.save(results_dict, os.path.join(output_folder, f"predictions_{device_id}.pth"))
    torch.save(gt_dict, os.path.join(output_folder, f"ground_truth_{device_id}.pth"))
    synchronize()


def aggregate_and_save_results(output_folder, world_size, config_dict):       
    results_dict_all = {}
    gt_dict_all = {}
    for i in range(world_size):
        results_dict = torch.load(os.path.join(output_folder, f"predictions_{i}.pth"))
        results_dict_all.update(results_dict)

        gt_dict = torch.load(os.path.join(output_folder, f"ground_truth_{i}.pth"))
        gt_dict_all.update(gt_dict)

    print(f"{len(results_dict_all.keys())} different frames")
    
    torch.save(results_dict_all, os.path.join(output_folder, "predictions_all.pth"))
    torch.save(gt_dict_all, os.path.join(output_folder, "ground_truth_all.pth"))
    with open(os.path.join(output_folder, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)
    return results_dict_all, gt_dict_all


def main(node_rank,
         local_rank,
         world_size,
         local_world_size,
         is_distributed,
         st_grouping_init_ckpt_path,
         st_grouping_ckpt_path,
         clip_pacl_ckpt_path,
         data_dir,
         dataset_img_dir,
         dataset_anno_dir, 
         img_index_path, 
         output_folder,
         num_slots,
         num_frames,
         vis,
         duplicate_box,
         n_stmae_seeds,
         n_spectral_clustering,
         refine_masks,
         filter_bg,
         merge_fg,
         n_batches,
         batch_offset):
    config_dict = locals()
    print(f"node_rank:{node_rank}")
    device_id = local_rank % local_world_size
    # init device
    nnodes = world_size // local_world_size
    if is_distributed:
        init_process_group()
    if torch.cuda.is_available():
        device = torch.device('cuda:{:d}'.format(local_rank % local_world_size))
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    # init dataset and data loader
    val_loader, val_dataset, denorm = init_data(is_distributed, device, data_dir, dataset_img_dir, dataset_anno_dir,
                                                img_index_path, num_frames)

    # init_model
    mae_grouping, clip_model, clip_pacl = init_model(device,
                                                     is_distributed,
                                                     num_slots,
                                                     num_frames,
                                                     st_grouping_init_ckpt_path,
                                                     st_grouping_ckpt_path,
                                                     clip_pacl_ckpt_path)
    
    prompts = Prompts_with_synonym(
        template_sentence_list,
        VID_classes,
        COCO_stuff_class,
        clip_model,
        device
    )
    foreground_prompts = prompts.generate_foreground_prompts()

    results_dict, gt_dict = do_inference(val_loader, mae_grouping, clip_pacl, denorm,
                                         prompts, foreground_prompts, vis, duplicate_box, device,
                                         num_frames, n_stmae_seeds, n_spectral_clustering, refine_masks,
                                         filter_bg, merge_fg, n_batches, batch_offset)
    synchronize()
    if is_distributed:
        timestamp = torch.zeros(1).long().cuda()
        if is_main_process():
            timestamp[0] = int(datetime.now().strftime("%m%d%H%M%S%f"))
            for i in range(1, world_size):
                dist.send(tensor=timestamp, dst=i)
        else:
            dist.recv(tensor=timestamp, src=0)
        synchronize()
        timestamp = str(timestamp.cpu().numpy()[0])
    else:
        timestamp = int(datetime.now().strftime("%m%d%m%H%M%S%f"))

    output_folder = f"{output_folder}_{timestamp}"
    save_results(output_folder, results_dict, gt_dict, device_id)
    if not is_main_process():
        return
    
    results_dict_all, gt_dict_all = aggregate_and_save_results(output_folder, world_size, config_dict)    
    print('Evaluating...')
    result = do_vid_evaluation(val_dataset, results_dict_all, gt_dict_all, output_folder)
    compute_unsupervised_metrics(results_dict_all, gt_dict_all, output_folder)

    return result

ds_path = ""

default_params = {
    "st_grouping_init_ckpt_path": ds_path+"data_ckpt_logs/ckpt/ssv2-single-frame-checkpoint-799.pth",
    "st_grouping_ckpt_path": ds_path+"data_ckpt_logs/ckpt/mae_grouping_299.pth",
    "clip_pacl_ckpt_path": ds_path+"data_ckpt_logs/ckpt/patch_based_clip.pth.tar",
    "data_dir": ds_path+"data_ckpt_logs/dataset/",
    "dataset_img_dir": ds_path+"data_ckpt_logs/dataset/ILSVRC/Data/VID",
    "dataset_anno_dir": ds_path+"data_ckpt_logs/dataset/ILSVRC/Annotations/VID",
    "img_index_path": ds_path+"data_ckpt_logs/dataset/ILSVRC/ImageSets/VID_val_videos.txt",
    "output_folder": "evaluation_results/auto_sc_ST",
    "num_slots": 15,
    "num_frames": 8,
    "vis": False,
    "duplicate_box": True,
    "n_stmae_seeds": 1,
    "n_spectral_clustering": [3, 10],
    "refine_masks": "None",
    "filter_bg": True,
    "merge_fg": True,
    "n_batches": -1,
    "batch_offset": 0,
    "seed": 287}

parser = argparse.ArgumentParser(description='')
for k, v in default_params.items():
    if isinstance(v, list):
        parser.add_argument(f"--{k}", default=v, type=type(v[0]), nargs='+')
    elif isinstance(v, bool):
        parser.add_argument(f"--{k}", default=str(v), type=str)
    else:
        parser.add_argument(f"--{k}", default=v, type=type(v))

if __name__ == '__main__':
    args = vars(parser.parse_args())
    params = copy.deepcopy(default_params)
    params.update(args)
    for k, v in default_params.items():
        if isinstance(v, bool) and not isinstance(args[k], bool):
            print(k, v, args[k])
            if args[k].lower() == "true":
                params[k] = True
            elif args[k].lower() == "false":
                params[k] = False
            else:
                raise ValueError(f"Unknown value for {k}")

    pprint(params)
    set_seed(params["seed"])
    del params["seed"]
    pprint(params)

    if os.environ.get("GROUP_RANK"):
        main(int(os.environ["GROUP_RANK"]),
             int(os.environ["LOCAL_RANK"]),
             int(os.environ["WORLD_SIZE"]),
             int(os.environ['LOCAL_WORLD_SIZE']),
             is_distributed=True,
             **params)
    else:
        main(0, 0, 1, 1, False,
             **params)
