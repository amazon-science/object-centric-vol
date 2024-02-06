import torch
from matplotlib import cm
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes


def _get_cmap(num_classes: int):
    if num_classes <= 20:
        mpl_cmap = cm.get_cmap("tab20", num_classes)(range(num_classes))
    else:
        mpl_cmap = cm.get_cmap("turbo", num_classes)(range(num_classes))
    cmap = [tuple((255 * cl[:3]).astype(int)) for cl in mpl_cmap]
    return cmap


def get_seg_vis_onehot(image: torch.Tensor, masks_one_hot: torch.Tensor):
    masks_one_hot = masks_one_hot.bool()
    input_image = image.permute(0, 3, 1, 2)
    # n_objects = len(masks_one_hot)
    n_objects = masks_one_hot.shape[1]

    cmap = _get_cmap(n_objects)
    masks_on_image = torch.stack(
        [
            draw_segmentation_masks(
                (255 * img).to(torch.uint8), mask, alpha=0.75, colors=cmap
            )
            for img, mask in zip(input_image.to("cpu"), masks_one_hot.to("cpu"))
        ]
    )

    return masks_on_image


def get_seg_vis(image: torch.Tensor, mask: torch.Tensor):
    assert image.dim() == 4

    input_image = image.permute(0, 3, 1, 2)
    n_objects = mask.shape[1]

    masks_argmax = mask.argmax(dim=1)[:, None]
    classes = torch.arange(n_objects)[None, :, None, None].to(masks_argmax)
    masks_one_hot = masks_argmax == classes

    cmap = _get_cmap(n_objects)
    masks_on_image = torch.stack(
        [
            draw_segmentation_masks(
                (255 * img).to(torch.uint8), mask, alpha=0.75, colors=cmap
            )
            for img, mask in zip(input_image.to("cpu"), masks_one_hot.to("cpu"))
        ]
    )

    return masks_on_image


def get_bbox_vis(images, bboxes, s_idx=0, e_idx=5):
    assert images.dim() == 4    # NHWC
    assert bboxes.dim() == 3    # NK4
    input_images = images.permute(0, 3, 1, 2)   # NCHW
    boxes_on_image = []
    for i in range(s_idx, e_idx):
        drawn_boxes = draw_bounding_boxes((255 * input_images[i]).to(torch.uint8),
                                          bboxes[i], colors='red')
        boxes_on_image.append(drawn_boxes)
    boxes_on_image = torch.stack(boxes_on_image, dim=0)

    return boxes_on_image


class Denormalize(torch.nn.Module):
    """Denormalize a tensor of shape (..., C, H, W) with any number of leading dimensions."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

    def __call__(self, tensor):
        return (tensor * self.std + self.mean).clamp(0.0, 1.0)

