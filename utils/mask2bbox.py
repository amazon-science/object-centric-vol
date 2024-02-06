import torch
import numpy as np
from .bounding_box import BoxList


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    n = masks.shape[0]

    bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

    for index, mask in enumerate(masks):
        y, x = torch.where(mask != 0)

        if x.size()[0] == 0:
            continue

        bounding_boxes[index, 0] = torch.min(x)
        bounding_boxes[index, 1] = torch.min(y)
        bounding_boxes[index, 2] = torch.max(x)
        bounding_boxes[index, 3] = torch.max(y)

    return bounding_boxes.cpu()


def convert_to_boxlist(boxes, category_predictions, height, width, duplicate_box=False):
    num_slots = len(category_predictions)
    assert num_slots == boxes.shape[0]

    if duplicate_box:
        num_classes = len(category_predictions[0])
        duplicated_boxes = torch.repeat_interleave(boxes, repeats=num_classes, dim=0)
        # duplicated_boxes = torch.cat([boxes for _ in range(num_classes)], dim=0)
        pred = BoxList(duplicated_boxes, (width, height), mode="xyxy")
        labels, scores = [], []
        for s in range(num_slots):
            for c in range(num_classes):
                labels.append(int(category_predictions[s][c][0]))
                scores.append(float(category_predictions[s][c][2]))
    else:
        pred = BoxList(boxes, (width, height), mode="xyxy")
        labels, scores = [], []
        for item in category_predictions:
            labels.append(int(item[0]))
            scores.append(float(item[2]))

    pred.add_field('labels', torch.from_numpy(np.asarray(labels)))
    pred.add_field('scores', torch.from_numpy(np.asarray(scores)))

    return pred


def get_image_bbox(mask):
    # input: mask [K, H, W]
    n_objects = mask.shape[0]       # K
    mask_argmax = mask.argmax(dim=0)[None, :]       # [1, H, W]
    classes = torch.arange(n_objects)[:, None, None].to(mask_argmax)        # [K, 1, 1]
    mask_one_hot = (mask_argmax == classes).long().to(mask.device)          # [K, H, W]
    boxes = masks_to_boxes(mask_one_hot)

    return boxes


def get_video_bbox(masks, num_frames):
    # input: masks [N, K, H, W]
    n_objects = masks.shape[1]          # K
    masks_argmax = masks.argmax(dim=1)[:, None]     # [N, 1, H, W]
    classes = torch.arange(n_objects)[None, :, None, None].to(masks_argmax)     # [1, K, 1, 1]
    masks_one_hot = (masks_argmax == classes).long().to(masks.device)           # [N, K, H, W]

    batched_boxes = []
    for frm in range(num_frames):
        boxes = masks_to_boxes(masks_one_hot[frm])
        batched_boxes.append(boxes)
    return batched_boxes