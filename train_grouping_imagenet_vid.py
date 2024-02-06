import os
import functools

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from models.model_grouping import GroupingVideoMAE
from utils.lr_scheduler import *
from visualization.vis_tools import *
from data.vid_dataset_train import build_pretraining_dataset_multistep_MultiResolution, BatchDataset
from pprint import pprint

import argparse
parser = argparse.ArgumentParser(description='ImageNet VID Video Object Grouping Training')

# for model
parser.add_argument('--n_slots', default=15, type=int,
                    help='number of slots')
parser.add_argument('--num_frames', default=8, type=int,
                    help='number of the video frames sampled')

# for data
parser.add_argument('--trainset', default="./data_ckpt_logs/dataset/ILSVRC2015_224px/train.csv", type=str,
                    help='path to train set')
parser.add_argument('--valset', default="./data_ckpt_logs/dataset/ILSVRC2015_224px/val.csv", type=str,
                    help='path to validation set')
parser.add_argument('--data_root', default="./data_ckpt_logs/dataset/ILSVRC2015_224px", type=str,
                    help='path to validation set')
parser.add_argument('-b', '--batch-size', default=7, type=int,
                    metavar='N',
                    help='mini-batch size , this is the batch size of each GPU')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--log_dir', default='./data_ckpt_logs/logs/logs_Grouping_ImageNetVID_VideoMAE_15slots', type=str,
                    help='path to validation set')
parser.add_argument('--ckpt_dir', default='./data_ckpt_logs/ckpt/checkpoint_Grouping_ImageNetVID_VideoMAE_15slots', type=str,
                    help='path to validation set')
parser.add_argument('--pretrained_checkpint', default='./data_ckpt_logs/ckpt/ssv2-single-frame-checkpoint-799.pth', type=str,
                    help='path to pretrained videomae')

# for optimizer
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')


def init_process_group():
    dist.init_process_group(
        backend='nccl',
        init_method='env://'),


def init_model(device, is_distributed, args):
    mae_grouping = GroupingVideoMAE(checkpoint_path = args.pretrained_checkpint,
                                    object_dim=128,
                                    n_slots=args.n_slots,
                                    feat_dim=768,
                                    num_patches=196,
                                    num_frames=args.num_frames,
                                    img_size=224).to(device)
    if is_distributed:
        if device.type == 'cpu':
            mae_grouping = DistributedDataParallel(mae_grouping)
        else:
            mae_grouping = DistributedDataParallel(mae_grouping, device_ids=[device], output_device=device)
    else:
        if device.type != 'cpu':
            mae_grouping = mae_grouping.to(device)

    model_params = list(mae_grouping.parameters())

    return mae_grouping, model_params


def main(node_rank, local_rank, world_size, local_world_size, is_distributed):
    args = parser.parse_args()
    if local_rank == 0:
        pprint(args)

    nnodes = world_size // local_world_size
    writer = SummaryWriter(log_dir=args.log_dir)
    ckpt_save_dir = args.ckpt_dir
    os.makedirs(ckpt_save_dir, exist_ok=True)

    if is_distributed:
        init_process_group()
    if torch.cuda.is_available():
        device = torch.device('cuda:{:d}'.format(local_rank % local_world_size))
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    train_dataset = build_pretraining_dataset_multistep_MultiResolution(args.trainset,
                                                                        224, args.num_frames, None,
                                                                        normalize=True, return_step=False, root=args.data_root)
    train_dataset = BatchDataset(train_dataset, batch_size=args.batch_size)
    val_dataset = build_pretraining_dataset_multistep_MultiResolution(args.valset,
                                                                      224, args.num_frames, None,
                                                                      normalize=True, return_step=False, root=args.data_root)
    denorm = Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)

    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
        
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler, num_workers=args.workers)

    mae_grouping, model_params = init_model(device, is_distributed, args=args)
    target_lr = args.lr
    optim = torch.optim.Adam(model_params, lr=target_lr)
    decay_fn = functools.partial(
        exp_decay_with_warmup_fn,
        decay_rate=0.5,
        decay_steps=50000,
        warmup_steps=5000,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, decay_fn)

    mae_grouping.train()

    epochs = args.epochs
    for epoch in range(epochs):
        train_loss = 0
        mae_grouping.train()
        for batch_id, batch in enumerate(train_loader):
            batch_dev = batch.to(device)[0]
            feat_recon_loss, masks_as_image = mae_grouping(batch_dev)
            loss = feat_recon_loss
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mae_grouping.parameters(), 1.0)
            optim.step()
            scheduler.step()
            train_loss += loss.item()
            if batch_id > 0 and batch_id % 10 == 0:
                print('Rank: %d, epoch: %d, batch: %03d / %d, loss: %.6f' % (
                local_rank, epoch, batch_id, len(train_loader), loss))
            if local_rank == 0:
                writer.add_scalar('Train/loss_feat_recon', loss.item(), batch_id + epoch * len(train_loader))
            if local_rank == 0 and batch_id > 0 and batch_id % 10 == 0:
                seg_img_pred = get_seg_vis(denorm(batch_dev[0]), masks_as_image[0])
                writer.add_image('Train/Raw_img', denorm(batch_dev[0]), batch_id + epoch * len(train_loader), dataformats='NHWC')
                writer.add_image('Train/Seg_img', seg_img_pred, batch_id + epoch * len(train_loader), dataformats='NCHW')

        torch.cuda.empty_cache()
        mae_grouping.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_id, batch in enumerate(val_loader):
                batch_dev = batch.to(device)
                feat_recon_loss, masks_as_image = mae_grouping(batch_dev)
                loss = feat_recon_loss
                val_loss += loss.item()
                if batch_id > 0 and batch_id % 10 == 0:
                    print('Testing Rank: %d, epoch: %d, batch: %03d / %d' % (
                        local_rank, epoch, batch_id, len(val_loader)))
                if node_rank == 0 and local_rank == 0:
                    if batch_id == 0:
                        writer.add_image('Validate/Raw_img', denorm(batch_dev[0]), epoch, dataformats='NHWC')
                        seg_img_pred = get_seg_vis(denorm(batch_dev[0]), masks_as_image[0])
                        writer.add_image('Validate/Seg_img', seg_img_pred, epoch, dataformats='NCHW')

        if local_rank == 0:
            writer.add_scalar('Validate/loss_feat_recon', val_loss / len(val_loader), epoch)
        print('Node Rank: %d, epoch: %d, train_loss: %.6f, val_loss :%.6f' % (
        node_rank, epoch, train_loss / len(train_loader), val_loss / len(val_loader)))

        if local_rank == 0 and (epoch + 1) % 20 == 0:
            torch.save(mae_grouping.module.state_dict(), os.path.join(ckpt_save_dir, 'mae_grouping_%03d.pth' % epoch))
        torch.cuda.empty_cache()

    dist.destroy_process_group()


if __name__ == '__main__':
    if os.environ.get("GROUP_RANK"):
        main(int(os.environ["GROUP_RANK"]), int(os.environ["LOCAL_RANK"]),
             int(os.environ["WORLD_SIZE"]), int(os.environ['LOCAL_WORLD_SIZE']),
             is_distributed=True)
    else:
        main(0, 0, 1, 1, False)

