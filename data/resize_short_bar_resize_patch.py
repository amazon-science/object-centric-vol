# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import os.path as osp
import sys
from multiprocessing import Pool
import tqdm
import time
import numpy as np

def resize_videos(vid_item, args):
    """Generate resized video cache.
    Args:
        vid_item (list): Video item containing video full path,
            video relative path.
    Returns:
        bool: Whether generate video cache successfully.
    """
    full_path, vid_path = vid_item
    # Change the output video extension to .mp4 if '--to-mp4' flag is set
    if args.to_mp4:
        vid_path =  vid_path[:-len(args.ext)] + 'mp4'
    out_full_path = osp.join(args.out_dir, vid_path)
    if os.path.exists(out_full_path):
        return
    dir_name = osp.dirname(vid_path)
    out_dir = osp.join(args.out_dir, dir_name)
    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    result = os.popen(
        f'ffprobe -hide_banner -loglevel error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 {full_path}'  # noqa:E501
    )
    try:
        w, h = [int(d) for d in result.readline().rstrip().split(',')]
        if w > h:
            l = w
            s = h
            new_l = int(np.round(l/s * args.scale / args.patch_size)) * args.patch_size
            cmd = (f'ffmpeg -hide_banner -loglevel error -i {full_path} '
                f'-vf {"mpdecimate," if args.remove_dup else ""}'
                f'scale={new_l}:{args.scale} '
                f'{"-vsync vfr" if args.remove_dup else ""} '
                f'-c:v libx264 {"-g 16" if args.dense else ""} '
                f'-an {out_full_path} -y')
            print(f"scale {w}:{h} -> {new_l}:{args.scale}")
        else:
            l = h
            s = w
            new_l = int(np.round(l/s * args.scale / args.patch_size)) * args.patch_size
            cmd = (f'ffmpeg -hide_banner -loglevel error -i {full_path} '
                f'-vf {"mpdecimate," if args.remove_dup else ""}'
                f'scale={args.scale}:{new_l} '
                f'{"-vsync vfr" if args.remove_dup else ""} '
                f'-c:v libx264 {"-g 16" if args.dense else ""} '
                f'-an {out_full_path} -y')
            print(f"scale {w}:{h} -> {args.scale}:{new_l}")
        os.popen(cmd)
        print(f'{vid_path} done')
        sys.stdout.flush()
        return True
    except:
        print(f'{vid_path} wrong')
        with open(args.corrupted_path, "a") as f:
            f.write(f'{vid_path} wrong\n')


def run_with_args(item):
    vid_item, args = item
    return resize_videos(vid_item, args)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate the resized cache of original videos')
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument('out_dir', type=str, help='output video directory')
    parser.add_argument(
        '--dense',
        action='store_true',
        help='whether to generate a faster cache')
    parser.add_argument(
        '--level',
        type=int,
        choices=[1, 2],
        default=2,
        help='directory level of data')
    parser.add_argument(
        '--remove-dup',
        action='store_true',
        help='whether to remove duplicated frames')
    parser.add_argument(
        '--ext',
        type=str,
        default='mp4',
        choices=['avi', 'mp4', 'webm', 'mkv'],
        help='video file extensions')
    parser.add_argument(
        '--to-mp4',
        action='store_true',
        help='whether to output videos in mp4 format')
    parser.add_argument(
        '--scale',
        type=int,
        default=240,
        help='resize image short side length keeping ratio')
    parser.add_argument(
        '--num-worker', type=int, default=32, help='number of workers')
    parser.add_argument(
        '--corrupted_path', type=str, default='corrupted.txt')
    parser.add_argument(
        '--patch_size', type=int, default=16)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not osp.isdir(args.out_dir):
        print(f'Creating folder: {args.out_dir}')
        os.makedirs(args.out_dir)

    print('Reading videos from folder: ', args.src_dir)
    print('Extension of videos: ', args.ext)
    fullpath_list = glob.glob(args.src_dir + '/*' * args.level + '.' +
                              args.ext)
    done_fullpath_list = glob.glob(args.out_dir + '/*' * args.level + args.ext)
    print('Total number of videos found: ', len(fullpath_list))
    print('Total number of videos transfer finished: ',
          len(done_fullpath_list))
    if args.level == 2:
        vid_list = list(
            map(
                lambda p: osp.join(
                    osp.basename(osp.dirname(p)), osp.basename(p)),
                fullpath_list))
    elif args.level == 1:
        vid_list = list(map(osp.basename, fullpath_list))
    
    vid_items = zip(fullpath_list, vid_list)
    vid_items_args = [(item, args) for item in vid_items]
    with Pool(args.num_worker) as pool:
        r = list(tqdm.tqdm(pool.imap(run_with_args, vid_items_args), total = len(vid_items_args)))
        