import os

import decord
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms


class Identity(object):
    def __call__(self, img_tuple):
        return img_tuple


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        return ([self.worker(img) for img in img_group], label)


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor_tuple):
        tensor, label = tensor_tuple
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return (tensor, label)


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_tuple):
        img_group, label = img_tuple

        if img_group[0].mode == "L":
            return (
                np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2),
                label,
            )
        elif img_group[0].mode == "RGB":
            if self.roll:
                return (
                    np.concatenate(
                        [np.array(x)[:, :, ::-1] for x in img_group], axis=2
                    ),
                    label,
                )
            else:
                return (np.concatenate(img_group, axis=2), label)


class ToTorchFormatTensor(object):
    """Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]"""

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic_tuple):
        pic, label = pic_tuple

        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return (img.float().div(255.0) if self.div else img.float(), label)


class DataAugmentationForVideoMAE(object):
    def __init__(self, input_size, bool_normalize=True, center_crop=True):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.center_crop = GroupCenterCrop(input_size) if center_crop else Identity()
        if bool_normalize:
            self.transform = transforms.Compose(
                [
                    self.center_crop,
                    Stack(roll=False),
                    ToTorchFormatTensor(div=True),
                    normalize,
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    self.center_crop,
                    Stack(roll=False),
                    ToTorchFormatTensor(div=True),
                ]
            )

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += ")"
        return repr


class BatchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        results = []
        for i in range(self.batch_size):
            results.append(self.dataset[index])
        return torch.stack(results)


class VideoDataset(torch.utils.data.Dataset):
    """Load your own video classification dataset.
    Parameters
    ----------
    root : str, required.
        Path to the root folder storing the dataset.
    setting : str, required.
        A text file describing the dataset, each line per video sample.
        There are three items in each line: (1) video path; (2) video length and (3) video label.
    train : bool, default True.
        Whether to load the training or validation set.
    test_mode : bool, default False.
        Whether to perform evaluation on the test set.
        Usually there is three-crop or ten-crop evaluation strategy involved.
    video_ext : str, default 'mp4'.
        If video_loader is set to True, please specify the video format accordinly.
    is_color : bool, default True.
        Whether the loaded image is color or grayscale.
    modality : str, default 'rgb'.
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    num_segments : int, default 1.
        Number of segments to evenly divide the video into clips. A useful technique to obtain
        global video-level information. Limin Wang, etal, Temporal Segment Networks: Towards Good
        Practices for Deep Action Recognition, ECCV 2016.
    num_crop : int, default 1.
        Number of crops for each image. default is 1.
        Common choices are three crops and ten crops during evaluation.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video
        frames. For example, new_length=16 means we will extract a video clip of consecutive 16
        frames.
    new_step : Union[int, Dict[int, float]]. Default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of
        consecutive frames; new_step=2 means we will extract a video clip of every other frame. Can
        also be a dict of steps to weights, in which case, every video will be sampled at a rate
        drawn randomly from the dict's keys with probabilities proportional to the dict's values.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    video_loader : bool, default False.
        Whether to use video loader to load data.
    transform : function, default None.
        A function that takes data and label and transforms them.
    data_aug : str, default 'v1'.
        Different types of data augmentation auto. Supports v1, v2, v3 and v4.
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
    """

    def __init__(
        self,
        root,
        setting,
        video_ext="mp4",
        num_segments=1,
        new_length=1,
        new_step=1,
        transform=None,
        temporal_jitter=False,
        video_loader=False,
        lazy_init=False,
        return_step=False,
        return_shape="THWC",
    ):
        super(VideoDataset, self).__init__()
        self.root = root
        self.setting = setting
        self.num_segments = num_segments
        self.new_length = new_length

        self.new_step_dict = {new_step: 1} if isinstance(new_step, int) else new_step
        self.new_step_list = np.array([i for i in self.new_step_dict.keys()])
        self.new_step_weight = np.array(
            [self.new_step_dict[i] for i in self.new_step_dict.keys()]
        )
        self.new_step_weight = self.new_step_weight / np.sum(self.new_step_weight)

        self.temporal_jitter = temporal_jitter
        self.video_loader = video_loader
        self.video_ext = video_ext
        self.transform = transform
        self.lazy_init = lazy_init
        self.return_step = return_step
        self.return_shape = return_shape

        if not self.lazy_init:
            self.clips = self._make_dataset(root, setting)
            if len(self.clips) == 0:
                raise (
                    RuntimeError(
                        "Found 0 video clips in subfolders of: " + root + "\n"
                        "Check your data directory (opt.data-dir)."
                    )
                )

    def __getitem__(self, index):
        directory, target = self.clips[index]
        if self.video_loader:
            if "." in directory.split("/")[-1]:
                # data in the "setting" file already have extension, e.g., demo.mp4
                video_name = directory
            else:
                # data in the "setting" file do not have extension, e.g., demo
                # So we need to provide extension (i.e., .mp4) to complete the file name.
                video_name = "{}.{}".format(directory, self.video_ext)

            decord_vr = decord.VideoReader(video_name, num_threads=1)
            duration = len(decord_vr)

        new_step = np.random.choice(self.new_step_list, p=self.new_step_weight)
        segment_indices, skip_offsets = self._sample_train_indices(duration, new_step)

        images = self._video_TSN_decord_batch_loader(
            directory, decord_vr, duration, segment_indices, skip_offsets, new_step
        )

        process_data = self.transform((images, None))  # T*C,H,W
        process_data = process_data.view(
            (self.new_length, 3) + process_data.size()[-2:]
        ).transpose(
            0, 1
        )  # T*C,H,W -> T,C,H,W -> C,T,H,W
        if self.return_shape == "THWC":
            process_data = process_data.permute(1, 2, 3, 0)
        elif self.return_shape == "CTHW":
            pass
        else:
            pass
        if self.return_step:
            return process_data, new_step
        else:
            return process_data

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self, directory, setting):
        if not os.path.exists(setting):
            raise (
                RuntimeError(
                    "Setting file %s doesn't exist. Check opt.train-list and opt.val-list. "
                    % (setting)
                )
            )
        dirname = self.root
        clips = []
        with open(setting) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split(" ")
                # line format: video_path, video_duration, video_label
                if len(line_info) < 2:
                    raise (
                        RuntimeError(
                            "Video input format is not correct, missing one or more element. %s"
                            % line
                        )
                    )
                clip_path = (
                    os.path.join(dirname, line_info[0])
                    if line_info[0][0] != "/"
                    else os.path.join(line_info[0])
                )
                target = int(line_info[1])
                item = (clip_path, target)
                clips.append(item)
        clips = sorted(clips)
        return clips

    def _sample_train_indices(self, num_frames, new_step):
        skip_length = self.new_length * new_step
        average_duration = (num_frames - skip_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration)
            offsets = offsets + np.random.randint(
                average_duration, size=self.num_segments
            )
        elif num_frames > max(self.num_segments, skip_length):
            offsets = np.sort(
                np.random.randint(num_frames - skip_length + 1, size=self.num_segments)
            )
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(new_step, size=skip_length // new_step)
        else:
            skip_offsets = np.zeros(skip_length // new_step, dtype=int)
        return offsets + 1, skip_offsets

    def _video_TSN_decord_batch_loader(
        self, directory, video_reader, duration, indices, skip_offsets, new_step
    ):
        skip_length = self.new_length * new_step
        sampled_list = []
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, skip_length, new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + new_step < duration:
                    offset += new_step
        try:
            decord.bridge.set_bridge("torch")
            video_data = video_reader.get_batch(frame_id_list)  # .asnumpy()

            # Ensure that both dimensions are dividable by 16 by resizing if required.
            # TODO: Improve this hack and test if image needs to be converted torch -> numpy -> PIL.
            _, h, w, c = video_data.shape
            if (h % 16 != 0) or (w % 16 != 0):
                w = int(round(w / 16.0) * 16)
                h = int(round(h / 16.0) * 16)
                video_data = transforms.functional.resize(
                    video_data.permute(0, 3, 1, 2), size=(h, w)
                ).permute(0, 2, 3, 1)

            video_data = video_data.numpy()
            sampled_list = [
                Image.fromarray(video_data[vid, :, :, :]).convert("RGB")
                for vid, _ in enumerate(frame_id_list)
            ]
        except Exception as e:
            raise RuntimeError(
                f"The following error occured in reading frames {frame_id_list} from video "
                f"{directory} of duration {duration}:\n{e}"
            )
        return sampled_list


def build_pretraining_dataset_multistep(
    data_path,
    input_size,
    num_frames,
    new_step,
    normalize=True,
    return_step=False,
    center_crop=True,
    root = None
):
    transform = DataAugmentationForVideoMAE(
        input_size, normalize, center_crop=center_crop
    )
    dataset = VideoDataset(
        root=root,
        setting=data_path,
        video_ext="mp4",
        new_length=num_frames,
        new_step=new_step,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        lazy_init=False,
        return_step=return_step,
    )
    print("Data Aug = %s" % str(transform))
    return dataset


def build_pretraining_dataset_multistep_MultiResolution(
    data_path, input_size, num_frames, step_dict=None, normalize=True, return_step=False, root = None
):
    center_crop = False
    step_dict = {1: 1, 2: 1, 4: 1, 8: 1} if step_dict is None else step_dict
    return build_pretraining_dataset_multistep(
        data_path,
        input_size,
        num_frames,
        step_dict,
        normalize,
        return_step,
        center_crop,
        root
    )
