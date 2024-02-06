from typing import Any, Dict, Optional
import os
import math
import numpy as np
import scipy.optimize
import torch
import torchmetrics
import torchvision


def resize_patches_to_image(
    patches: torch.Tensor,
    size: Optional[int] = None,
    scale_factor: Optional[float] = None,
    resize_mode: str = "bilinear",
) -> torch.Tensor:
    """Convert and resize a tensor of patches to image shape.

    This method requires that the patches can be converted to a square image.

    Args:
        patches: Patches to be converted of shape (..., C, P), where C is the number of channels and
            P the number of patches.
        size: Image size to resize to.
        scale_factor: Scale factor by which to resize the patches. Can be specified alternatively to
            `size`.
        resize_mode: Method to resize with. Valid options are "nearest", "nearest-exact", "bilinear",
            "bicubic".

    Returns: Tensor of shape (..., C, S, S) where S is the image size.
    """
    has_size = size is None
    has_scale = scale_factor is None
    if has_size == has_scale:
        raise ValueError("Exactly one of `size` or `scale_factor` must be specified.")

    n_channels = patches.shape[-2]
    n_patches = patches.shape[-1]
    patch_size_float = math.sqrt(n_patches)
    patch_size = int(math.sqrt(n_patches))
    if patch_size_float != patch_size:
        raise ValueError("The number of patches needs to be a perfect square.")

    image = torch.nn.functional.interpolate(
        patches.view(-1, n_channels, patch_size, patch_size),
        size=size,
        scale_factor=scale_factor,
        mode=resize_mode,
    )

    return image.view(*patches.shape[:-1], image.shape[-2], image.shape[-1])


def tensor_to_one_hot(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Convert tensor to one-hot encoding by using maximum across dimension as one-hot element."""
    assert 0 <= dim
    max_idxs = torch.argmax(tensor, dim=dim, keepdim=True)
    shape = [1] * dim + [-1] + [1] * (tensor.ndim - dim - 1)
    one_hot = max_idxs == torch.arange(tensor.shape[dim], device=tensor.device).view(*shape)
    return one_hot.to(torch.long)


class TensorStatistic(torchmetrics.Metric):
    """Metric that computes summary statistic of tensors for logging purposes.

    First dimension of tensor is assumed to be batch dimension. Other dimensions are reduced to a
    scalar by the chosen reduction approach (sum or mean).
    """

    def __init__(self, path: Optional[str], reduction: str = "mean"):
        torchmetrics.Metric.__init__(self)
        if reduction not in ("sum", "mean"):
            raise ValueError(f"Unknown reduction {reduction}")
        self.reduction = reduction
        self.add_state(
            "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, tensor: torch.Tensor):
        tensor = torch.atleast_2d(tensor).flatten(1, -1).to(dtype=torch.float64)

        if self.reduction == "mean":
            tensor = torch.mean(tensor, dim=1)
        elif self.reduction == "sum":
            tensor = torch.sum(tensor, dim=1)

        self.values += tensor.sum()
        self.total += len(tensor)

    def compute(self) -> torch.Tensor:
        return self.values / self.total


class TorchmetricsWrapper(torchmetrics.Metric):
    """Wrapper for torchmetrics classes that works with routing."""

    def __init__(
        self,
        metric: str,
        prediction_path: str,
        target_path: str,
        metric_kwargs: Optional[Dict[str, Any]] = None,
    ):
        torchmetrics.Metric.__init__(self)
        if not hasattr(torchmetrics, metric):
            raise ValueError(f"Metric {metric} does not exist in torchmetrics")
        self.metric = getattr(torchmetrics, metric)(**(metric_kwargs if metric_kwargs else {}))

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        return self.metric.update(prediction, target)

    def compute(self) -> torch.Tensor:
        return self.metric.compute()


class ARIMetric(torchmetrics.Metric):
    """Computes ARI metric."""

    def __init__(
        self,
        foreground: bool = True,
        convert_target_one_hot: bool = False,
    ):
        torchmetrics.Metric.__init__(self)
        self.foreground = foreground
        self.convert_target_one_hot = convert_target_one_hot
        self.add_state("values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        """Update this metric.

        Args:
            prediction: Predicted mask of shape (B, C, H, W) or (B, F, C, H, W), where C is the
                number of classes.
            target: Ground truth mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the
                number of classes.
        """
        if prediction.ndim == 5:
            # Merge frames, height and width to single dimension.
            prediction = prediction.transpose(1, 2).flatten(-3, -1)
            target = target.transpose(1, 2).flatten(-3, -1)
        elif prediction.ndim == 4:
            # Merge height and width to single dimension.
            prediction = prediction.flatten(-2, -1)
            target = target.flatten(-2, -1)
        else:
            raise ValueError(f"Incorrect input shape: f{prediction.shape}")

        # Make channels / gt labels the last dimension.
        prediction = prediction.transpose(-2, -1)
        target = target.transpose(-2, -1)

        if self.convert_target_one_hot:
            target_oh = tensor_to_one_hot(target, dim=2)
            # For empty pixels (all values zero), one-hot assigns 1 to the first class, correct for
            # this (then it is technically not one-hot anymore).
            target_oh[:, :, 0][target.sum(dim=2) == 0] = 0
            target = target_oh

        # Should be either 0 (empty, padding) or 1 (single object).
        assert torch.all(target.sum(dim=-1) < 2), "Issues with target format, mask non-exclusive"

        if self.foreground:
            ari = fg_adjusted_rand_index(prediction, target)
        else:
            ari = adjusted_rand_index(prediction, target)

        self.values += ari.sum()
        self.total += len(ari)

    def compute(self) -> torch.Tensor:
        return self.values / self.total


class PatchARIMetric(ARIMetric):
    """Computes ARI metric assuming patch masks as input."""

    def __init__(
        self,
        prediction_key: str,
        target_key: str,
        foreground=True,
        resize_masks_mode: str = "bilinear",
        **kwargs,
    ):
        super().__init__(prediction_key, target_key, foreground, **kwargs)
        self.resize_masks_mode = resize_masks_mode

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        """Update this metric.

        Args:
            prediction: Predicted mask of shape (B, C, P) or (B, F, C, P), where C is the
                number of classes and P the number of patches.
            target: Ground truth mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the
                number of classes.
        """
        h, w = target.shape[-2:]
        assert h == w

        prediction_resized = resize_patches_to_image(
            prediction, size=h, resize_mode=self.resize_masks_mode
        )

        return super().update(prediction=prediction_resized, target=target)


def adjusted_rand_index(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
    """Computes adjusted Rand index (ARI), a clustering similarity score.

    This implementation ignores points with no cluster label in `true_mask` (i.e. those points for
    which `true_mask` is a zero vector). In the context of segmentation, that means this function
    can ignore points in an image corresponding to the background (i.e. not to an object).

    Implementation adapted from https://github.com/deepmind/multi_object_datasets and
    https://github.com/google-research/slot-attention-video/blob/main/savi/lib/metrics.py

    Args:
        pred_mask: Predicted cluster assignment encoded as categorical probabilities of shape
            (batch_size, n_points, n_pred_clusters).
        true_mask: True cluster assignment encoded as one-hot of shape (batch_size, n_points,
            n_true_clusters).

    Returns:
        ARI scores of shape (batch_size,).
    """
    n_pred_clusters = pred_mask.shape[-1]
    pred_cluster_ids = torch.argmax(pred_mask, axis=-1)

    # Convert true and predicted clusters to one-hot ('oh') representations. We use float64 here on
    # purpose, otherwise mixed precision training automatically casts to FP16 in some of the
    # operations below, which can create overflows.
    true_mask_oh = true_mask.to(torch.float64)  # already one-hot
    pred_mask_oh = torch.nn.functional.one_hot(pred_cluster_ids, n_pred_clusters).to(torch.float64)

    n_ij = torch.einsum("bnc,bnk->bck", true_mask_oh, pred_mask_oh)
    a = torch.sum(n_ij, axis=-1)
    b = torch.sum(n_ij, axis=-2)
    n_fg_points = torch.sum(a, axis=1)

    rindex = torch.sum(n_ij * (n_ij - 1), axis=(1, 2))
    aindex = torch.sum(a * (a - 1), axis=1)
    bindex = torch.sum(b * (b - 1), axis=1)
    expected_rindex = aindex * bindex / torch.clamp(n_fg_points * (n_fg_points - 1), min=1)
    max_rindex = (aindex + bindex) / 2
    denominator = max_rindex - expected_rindex
    ari = (rindex - expected_rindex) / denominator

    # There are two cases for which the denominator can be zero:
    # 1. If both true_mask and pred_mask assign all pixels to a single cluster.
    #    (max_rindex == expected_rindex == rindex == n_fg_points * (n_fg_points-1))
    # 2. If both true_mask and pred_mask assign max 1 point to each cluster.
    #    (max_rindex == expected_rindex == rindex == 0)
    # In both cases, we want the ARI score to be 1.0:
    return torch.where(denominator > 0, ari, torch.ones_like(ari))


def fg_adjusted_rand_index(
    pred_mask: torch.Tensor, true_mask: torch.Tensor, bg_dim: int = 0
) -> torch.Tensor:
    """Compute adjusted random index using only foreground groups (FG-ARI).

    Args:
        pred_mask: Predicted cluster assignment encoded as categorical probabilities of shape
            (batch_size, n_points, n_pred_clusters).
        true_mask: True cluster assignment encoded as one-hot of shape (batch_size, n_points,
            n_true_clusters).
        bg_dim: Index of background class in true mask.

    Returns:
        ARI scores of shape (batch_size,).
    """
    n_true_clusters = true_mask.shape[-1]
    assert 0 <= bg_dim < n_true_clusters
    if bg_dim == 0:
        true_mask_only_fg = true_mask[..., 1:]
    elif bg_dim == n_true_clusters - 1:
        true_mask_only_fg = true_mask[..., :-1]
    else:
        true_mask_only_fg = torch.cat(
            (true_mask[..., :bg_dim], true_mask[..., bg_dim + 1 :]), dim=-1
        )

    return adjusted_rand_index(pred_mask, true_mask_only_fg)


def _all_equal_masked(values: torch.Tensor, mask: torch.Tensor, dim=-1) -> torch.Tensor:
    """Check if all masked values along a dimension of a tensor are the same.

    All non-masked values are considered as true, i.e. if no value is masked, true is returned
    for this dimension.
    """
    assert mask.dtype == torch.bool
    _, first_non_masked_idx = torch.max(mask, dim=dim)

    comparison_value = values.gather(index=first_non_masked_idx.unsqueeze(dim), dim=dim)

    return torch.logical_or(~mask, values == comparison_value).all(dim=dim)


class UnsupervisedMaskIoUMetric(torchmetrics.Metric):
    """Computes IoU metric for segmentation masks when correspondences to ground truth are not known.

    Uses Hungarian matching to compute the assignment between predicted classes and ground truth
    classes.

    Args:
        use_threshold: If `True`, convert predicted class probabilities to mask using a threshold.
            If `False`, class probabilities are turned into mask using a softmax instead.
        threshold: Value to use for thresholding masks.
        matching: Approach to match predicted to ground truth classes. For "hungarian", computes
            assignment that maximizes total IoU between all classes. For "best_overlap", uses the
            predicted class with maximum overlap for each ground truth class. Using "best_overlap"
            leads to the "average best overlap" metric.
        compute_discovery_fraction: Instead of the IoU, compute the fraction of ground truth classes
            that were "discovered", meaning that they have an IoU greater than some threshold.
        correct_localization: Instead of the IoU, compute the fraction of images on which at least
            one ground truth class was correctly localised, meaning that they have an IoU
            greater than some threshold.
        discovery_threshold: Minimum IoU to count a class as discovered/correctly localized.
        ignore_background: If true, assume class at index 0 of ground truth masks is background class
            that is removed before computing IoU.
    """

    def __init__(
        self,
        prediction_path: str,
        target_path: str,
        use_threshold: bool = False,
        threshold: float = 0.5,
        matching: str = "hungarian",
        compute_discovery_fraction: bool = False,
        correct_localization: bool = False,
        discovery_threshold: float = 0.5,
        ignore_background: bool = False,
    ):
        torchmetrics.Metric.__init__(self)
        self.use_threshold = use_threshold
        self.threshold = threshold
        self.discovery_threshold = discovery_threshold
        self.compute_discovery_fraction = compute_discovery_fraction
        self.correct_localization = correct_localization
        if compute_discovery_fraction and correct_localization:
            raise ValueError(
                "Only one of `compute_discovery_fraction` and `correct_localization` can be enabled."
            )

        matchings = ("hungarian", "best_overlap")
        if matching not in matchings:
            raise ValueError(f"Unknown matching type {matching}. Valid values are {matchings}.")
        self.matching = matching
        self.ignore_background = ignore_background

        self.add_state(
            "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        """Update this metric.

        Args:
            prediction: Predicted mask of shape (B, C, H, W) or (B, F, C, H, W), where C is the
                number of classes. Assumes class probabilities as inputs.
            target: Ground truth mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the
                number of classes.
        """
        if prediction.ndim == 5:
            # Merge frames, height and width to single dimension.
            predictions = prediction.transpose(1, 2).flatten(-3, -1)
            targets = target.transpose(1, 2).flatten(-3, -1)
        elif prediction.ndim == 4:
            # Merge height and width to single dimension.
            predictions = prediction.flatten(-2, -1)
            targets = target.flatten(-2, -1)
        else:
            raise ValueError(f"Incorrect input shape: f{prediction.shape}")

        if self.use_threshold:
            predictions = predictions > self.threshold
        else:
            indices = torch.argmax(predictions, dim=1)
            predictions = torch.nn.functional.one_hot(indices, num_classes=predictions.shape[1])
            predictions = predictions.transpose(1, 2)

        if self.ignore_background:
            target = target[:, 1:]

        for pred, target in zip(predictions, targets):
            nonzero_classes = torch.nonzero(torch.sum(target, dim=-1) > 0).view(-1)
            if len(nonzero_classes) == 0:
                continue  # Skip elements without any target mask
            n_gt_classes = torch.max(nonzero_classes) + 1
            target = target[:n_gt_classes]  # Remove padded classes

            iou_per_class = unsupervised_mask_iou(
                pred, target, matching=self.matching, reduction="none"
            )
            if self.compute_discovery_fraction:
                discovered = iou_per_class > self.discovery_threshold
                self.values += discovered.sum() / len(discovered)
            elif self.correct_localization:
                correctly_localized = torch.any(iou_per_class > self.discovery_threshold)
                self.values += correctly_localized.sum()
            else:
                self.values += iou_per_class.mean()
            self.total += 1

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.zeros_like(self.values)
        else:
            return self.values / self.total


def unsupervised_mask_iou(
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
    matching: str = "hungarian",
    reduction: str = "mean",
    iou_empty: float = 0.0,
) -> torch.Tensor:
    """Compute intersection-over-union (IoU) between masks with unknown class correspondences.

    This metric is also known as Jaccard index. Note that this is a non-batched implementation.

    Args:
        pred_mask: Predicted mask of shape (C, N), where C is the number of predicted classes and
            N is the number of points. Masks are assumed to be binary.
        true_mask: Ground truth mask of shape (K, N), where K is the number of ground truth
            classes and N is the number of points. Masks are assumed to be binary.
        matching: How to match predicted classes to ground truth classes. For "hungarian", computes
            assignment that maximizes total IoU between all classes. For "best_overlap", uses the
            predicted class with maximum overlap for each ground truth class (each predicted class
            can be assigned to multiple ground truth classes). Empty ground truth classes are
            assigned IoU of zero.
        reduction: If "mean", return IoU averaged over classes. If "none", return per-class IoU.
        iou_empty: IoU for the case when a class does not occur, but was also not predicted.

    Returns:
        Mean IoU over classes if reduction is `mean`, tensor of shape (K,) containing per-class IoU
        otherwise.
    """
    n_gt_classes = len(true_mask)
    pred_mask = pred_mask.unsqueeze(1).to(torch.bool)
    true_mask = true_mask.unsqueeze(0).to(torch.bool)

    intersection = torch.sum(pred_mask & true_mask, dim=-1).to(torch.float64)
    union = torch.sum(pred_mask | true_mask, dim=-1).to(torch.float64)
    pairwise_iou = intersection / union
    # Remove NaN from divide-by-zero: class does not occur, and class was not predicted.
    pairwise_iou[union == 0] = iou_empty

    if matching == "hungarian":
        pred_idxs, true_idxs = scipy.optimize.linear_sum_assignment(
            pairwise_iou.cpu(), maximize=True
        )
        pred_idxs = torch.as_tensor(pred_idxs, dtype=torch.int64, device=pairwise_iou.device)
        true_idxs = torch.as_tensor(true_idxs, dtype=torch.int64, device=pairwise_iou.device)
    elif matching == "best_overlap":
        non_empty_gt = torch.sum(true_mask.squeeze(0), dim=1) > 0
        pred_idxs = torch.argmax(pairwise_iou, dim=0)[non_empty_gt]
        true_idxs = torch.arange(pairwise_iou.shape[1])[non_empty_gt]
    else:
        raise ValueError(f"Unknown matching {matching}")

    matched_iou = pairwise_iou[pred_idxs, true_idxs]
    iou = torch.zeros(n_gt_classes, dtype=torch.float64, device=pairwise_iou.device)
    iou[true_idxs] = matched_iou

    if reduction == "mean":
        return iou.mean()
    else:
        return iou


class UnsupervisedBboxIoUMetric(torchmetrics.Metric):
    """Computes IoU metric for bounding boxes when correspondences to ground truth are not known.

    Currently, assumes segmentation masks as input for both prediction and targets.

    Args:
        target_is_mask: If `True`, assume input is a segmentation mask, in which case the masks are
            converted to bounding boxes before computing IoU. If `False`, assume the input for the
            targets are already bounding boxes.
        use_threshold: If `True`, convert predicted class probabilities to mask using a threshold.
            If `False`, class probabilities are turned into mask using a softmax instead.
        threshold: Value to use for thresholding masks.
        matching: How to match predicted boxes to ground truth boxes. For "hungarian", computes
            assignment that maximizes total IoU between all boxes. For "best_overlap", uses the
            predicted box with maximum overlap for each ground truth box (each predicted box
            can be assigned to multiple ground truth boxes).
        compute_discovery_fraction: Instead of the IoU, compute the fraction of ground truth classes
            that were "discovered", meaning that they have an IoU greater than some threshold. This
            is recall, or sometimes called the detection rate metric.
        correct_localization: Instead of the IoU, compute the fraction of images on which at least
            one ground truth bounding box was correctly localised, meaning that they have an IoU
            greater than some threshold.
        discovery_threshold: Minimum IoU to count a class as discovered/correctly localized.
    """

    def __init__(
        self,
        prediction_path: str,
        target_path: str,
        target_is_mask: bool = False,
        use_threshold: bool = False,
        threshold: float = 0.5,
        matching: str = "hungarian",
        compute_discovery_fraction: bool = False,
        correct_localization: bool = False,
        discovery_threshold: float = 0.5,
    ):
        torchmetrics.Metric.__init__(self)
        self.target_is_mask = target_is_mask
        self.use_threshold = use_threshold
        self.threshold = threshold
        self.discovery_threshold = discovery_threshold
        self.compute_discovery_fraction = compute_discovery_fraction
        self.correct_localization = correct_localization
        if compute_discovery_fraction and correct_localization:
            raise ValueError(
                "Only one of `compute_discovery_fraction` and `correct_localization` can be enabled."
            )

        matchings = ("hungarian", "best_overlap")
        if matching not in matchings:
            raise ValueError(f"Unknown matching type {matching}. Valid values are {matchings}.")
        self.matching = matching

        self.add_state(
            "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        """Update this metric.

        Args:
            prediction: Predicted mask of shape (B, C, H, W) or (B, F, C, H, W), where C is the
                number of instances. Assumes class probabilities as inputs.
            target: Ground truth mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the
                number of instance, if using masks as input, or bounding boxes of shape (B, K, 4)
                or (B, F, K, 4).
        """
        if prediction.ndim == 5:
            # Merge batch and frame dimensions
            prediction = prediction.flatten(0, 1)
            target = target.flatten(0, 1)
        elif prediction.ndim != 4:
            raise ValueError(f"Incorrect input shape: f{prediction.shape}")

        bs, n_pred_classes = prediction.shape[:2]
        n_gt_classes = target.shape[1]

        if self.use_threshold:
            prediction = prediction > self.threshold
        else:
            indices = torch.argmax(prediction, dim=1)
            prediction = torch.nn.functional.one_hot(indices, num_classes=n_pred_classes)
            prediction = prediction.permute(0, 3, 1, 2)

        pred_bboxes = masks_to_bboxes(prediction.flatten(0, 1)).unflatten(0, (bs, n_pred_classes))
        if self.target_is_mask:
            target_bboxes = masks_to_bboxes(target.flatten(0, 1)).unflatten(0, (bs, n_gt_classes))
        else:
            assert target.shape[-1] == 4
            # Convert all-zero boxes added during padding to invalid boxes
            target[torch.all(target == 0.0, dim=-1)] = -1.0
            target_bboxes = target

        for pred, target in zip(pred_bboxes, target_bboxes):
            valid_pred_bboxes = pred[:, 0] != -1.0
            valid_target_bboxes = target[:, 0] != -1.0
            if valid_target_bboxes.sum() == 0:
                continue  # Skip data points without any target bbox

            pred = pred[valid_pred_bboxes]
            target = target[valid_target_bboxes]

            if valid_pred_bboxes.sum() > 0:
                iou_per_bbox = unsupervised_bbox_iou(
                    pred, target, matching=self.matching, reduction="none"
                )
            else:
                iou_per_bbox = torch.zeros_like(valid_target_bboxes, dtype=torch.float32)

            if self.compute_discovery_fraction:
                discovered = iou_per_bbox > self.discovery_threshold
                self.values += discovered.sum() / len(iou_per_bbox)
            elif self.correct_localization:
                correctly_localized = torch.any(iou_per_bbox > self.discovery_threshold)
                self.values += correctly_localized.sum()
            else:
                self.values += iou_per_bbox.mean()
            self.total += 1

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.zeros_like(self.values)
        else:
            return self.values / self.total


class UnsupervisedBboxIoUMetricMinimal(torchmetrics.Metric):
    """Computes IoU metric for bounding boxes when correspondences to ground truth are not known.

    Assumes bounding boxes as input for both prediction and targets.

    Args:
        matching: How to match predicted boxes to ground truth boxes. For "hungarian", computes
            assignment that maximizes total IoU between all boxes. For "best_overlap", uses the
            predicted box with maximum overlap for each ground truth box (each predicted box
            can be assigned to multiple ground truth boxes).
        compute_discovery_fraction: Instead of the IoU, compute the fraction of ground truth classes
            that were "discovered", meaning that they have an IoU greater than some threshold. This
            is recall, or sometimes called the detection rate metric.
        correct_localization: Instead of the IoU, compute the fraction of images on which at least
            one ground truth bounding box was correctly localised, meaning that they have an IoU
            greater than some threshold.
        discovery_threshold: Minimum IoU to count a class as discovered/correctly localized.
    """

    def __init__(
        self,
        matching: str = "hungarian",
        compute_discovery_fraction: bool = False,
        correct_localization: bool = False,
        discovery_threshold: float = 0.5,
    ):
        torchmetrics.Metric.__init__(self)
        self.discovery_threshold = discovery_threshold
        self.compute_discovery_fraction = compute_discovery_fraction
        self.correct_localization = correct_localization
        if compute_discovery_fraction and correct_localization:
            raise ValueError(
                "Only one of `compute_discovery_fraction` and `correct_localization` can be enabled."
            )

        matchings = ("hungarian", "best_overlap")
        if matching not in matchings:
            raise ValueError(f"Unknown matching type {matching}. Valid values are {matchings}.")
        self.matching = matching

        self.add_state(
            "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_bboxes, target_bboxes):
        """Update this metric.
        Args:
            pred_bboxes: Predicted bounding boxes of shape (B, K, 4).
            target_bboxes: bounding boxes of shape (B, K, 4).
        """
        for pred, target in zip(pred_bboxes, target_bboxes):
            valid_pred_bboxes = pred[:, 0] != -1.0
            valid_target_bboxes = target[:, 0] != -1.0
            if valid_target_bboxes.sum() == 0:
                continue  # Skip data points without any target bbox

            pred = pred[valid_pred_bboxes]
            target = target[valid_target_bboxes]

            if valid_pred_bboxes.sum() > 0:
                iou_per_bbox = unsupervised_bbox_iou(
                    pred, target, matching=self.matching, reduction="none"
                )
            else:
                iou_per_bbox = torch.zeros_like(valid_target_bboxes, dtype=torch.float32)

            if self.compute_discovery_fraction:
                discovered = iou_per_bbox > self.discovery_threshold
                self.values += discovered.sum() / len(iou_per_bbox)
            elif self.correct_localization:
                correctly_localized = torch.any(iou_per_bbox > self.discovery_threshold)
                self.values += correctly_localized.sum()
            else:
                self.values += iou_per_bbox.mean()
            self.total += 1

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.zeros_like(self.values)
        else:
            return self.values / self.total


def unsupervised_bbox_iou(
    pred_bboxes: torch.Tensor,
    true_bboxes: torch.Tensor,
    matching: str = "best_overlap",
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute IoU between two sets of bounding boxes.

    Args:
        pred_bboxes: Predicted bounding boxes of shape N x 4.
        true_bboxes: True bounding boxes of shape M x 4.
        matching: Method to assign predicted to true bounding boxes.
        reduction: Whether to average the computes IoUs per true box.
    """
    n_gt_bboxes = len(true_bboxes)

    pairwise_iou = torchvision.ops.box_iou(pred_bboxes, true_bboxes)

    if matching == "hungarian":
        pred_idxs, true_idxs = scipy.optimize.linear_sum_assignment(
            pairwise_iou.cpu(), maximize=True
        )
        pred_idxs = torch.as_tensor(pred_idxs, dtype=torch.int64, device=pairwise_iou.device)
        true_idxs = torch.as_tensor(true_idxs, dtype=torch.int64, device=pairwise_iou.device)
    elif matching == "best_overlap":
        pred_idxs = torch.argmax(pairwise_iou, dim=0)
        true_idxs = torch.arange(pairwise_iou.shape[1], device=pairwise_iou.device)
    else:
        raise ValueError(f"Unknown matching {matching}")

    matched_iou = pairwise_iou[pred_idxs, true_idxs]

    iou = torch.zeros(n_gt_bboxes, dtype=torch.float32, device=pairwise_iou.device)
    iou[true_idxs] = matched_iou

    if reduction == "mean":
        return iou.mean()
    else:
        return iou


def masks_to_bboxes(masks: torch.Tensor, empty_value: float = -1.0) -> torch.Tensor:
    """Compute bounding boxes around the provided masks.

    Adapted from DETR: https://github.com/facebookresearch/detr/blob/main/util/box_ops.py

    Args:
        masks: Tensor of shape (N, H, W), where N is the number of masks, H and W are the spatial
            dimensions.
        empty_value: Value bounding boxes should contain for empty masks.

    Returns:
        Tensor of shape (N, 4), containing bounding boxes in (x1, y1, x2, y2) format, where (x1, y1)
        is the coordinate of top-left corner and (x2, y2) is the coordinate of the bottom-right
        corner (inclusive) in pixel coordinates. If mask is empty, all coordinates contain
        `empty_value` instead.
    """
    masks = masks.bool()
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    large_value = 1e8
    inv_mask = ~masks

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    y, x = torch.meshgrid(y, x, indexing="ij")

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(inv_mask, large_value).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(inv_mask, large_value).flatten(1).min(-1)[0]

    bboxes = torch.stack((x_min, y_min, x_max, y_max), dim=1)
    bboxes[x_min == large_value] = empty_value

    return bboxes


class DatasetSemanticMaskIoUMetric(torchmetrics.Metric):
    """Unsupervised IoU metric for semantic segmentation using dataset-wide matching of classes.

    The input to this metric is an instance-level mask with objects, and a class id for each object.
    This is required to convert the mask to semantic classes. The number of classes for the
    predictions does not have to match the true number of classes.

    Note that contrary to the other metrics in this module, this metric is not supposed to be added
    in the online metric computation loop, which is why it does not inherit from `RoutableMixin`.

    Args:
        n_predicted_classes: Number of predictable classes, i.e. highest prediction class id that can
            occur.
        n_classes: Total number of classes, i.e. highest class id that can occur.
        threshold: Value to use for thresholding masks.
        use_threshold: If `True`, convert predicted class probabilities to mask using a threshold.
            If `False`, class probabilities are turned into mask using an argmax instead.
        matching: Method to produce matching between clusters and ground truth classes. If
            "hungarian", assigns each class one cluster such that the total IoU is maximized. If
            "majority", assigns each cluster to the class with the highest IoU (each class can be
            assigned multiple clusters).
        ignore_background: If true, pixels labeled as background (class zero) in the ground truth
            are not taken into account when computing IoU.
        use_unmatched_as_background: If true, count predicted classes not selected after Hungarian
            matching as the background predictions.
    """

    def __init__(
        self,
        n_predicted_classes: int,
        n_classes: int,
        use_threshold: bool = False,
        threshold: float = 0.5,
        matching: str = "hungarian",
        ignore_background: bool = False,
        use_unmatched_as_background: bool = False,
    ):
        super().__init__()
        matching_methods = {"hungarian", "majority"}
        if matching not in matching_methods:
            raise ValueError(
                f"Unknown matching method {matching}. Valid values are {matching_methods}."
            )

        self.matching = matching
        self.n_predicted_classes = n_predicted_classes
        self.n_predicted_classes_with_bg = n_predicted_classes + 1
        self.n_classes = n_classes
        self.n_classes_with_bg = n_classes + 1
        self.matching = matching
        self.use_threshold = use_threshold
        self.threshold = threshold
        self.ignore_background = ignore_background
        self.use_unmatched_as_background = use_unmatched_as_background
        if use_unmatched_as_background and ignore_background:
            raise ValueError(
                "Option `use_unmatched_as_background` not compatible with option `ignore_background`"
            )
        if use_unmatched_as_background and matching == "majority":
            raise ValueError(
                "Option `use_unmatched_as_background` not compatible with matching `majority`"
            )

        confusion_mat = torch.zeros(
            self.n_predicted_classes_with_bg, self.n_classes_with_bg, dtype=torch.int64
        )
        self.add_state("confusion_mat", default=confusion_mat, dist_reduce_fx="sum", persistent=True)

    def update(
        self, predictions: torch.Tensor, targets: torch.Tensor, prediction_class_ids: torch.Tensor
    ):
        """Update metric by computing confusion matrix between predicted and target classes.

        Args:
            predictions: Probability mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the
                number of object instances in the image.
            targets: Mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the number of object
                instances in the image. Class ID of objects is encoded as the value, i.e. densely
                represented.
            prediction_class_ids: Tensor of shape (B, K), containing the class id of each predicted
                object instance in the image. Id must be 0 <= id <= n_predicted_classes.
        """
        predictions = self.preprocess_predicted_mask(predictions)
        predictions = _remap_one_hot_mask(
            predictions, prediction_class_ids, self.n_predicted_classes, strip_empty=False
        )
        assert predictions.shape[-1] == self.n_predicted_classes_with_bg

        targets = self.preprocess_ground_truth_mask(targets)
        assert targets.shape[-1] == self.n_classes_with_bg

        # We are doing the multiply in float64 instead of int64 because it proved to be significantly
        # faster on GPU. We need to use 64 bits because we can easily exceed the range of 32 bits
        # if we aggregate over a full dataset.
        confusion_mat = torch.einsum(
            "bpk,bpc->kc", predictions.to(torch.float64), targets.to(torch.float64)
        )
        self.confusion_mat += confusion_mat.to(torch.int64)

    def preprocess_predicted_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Preprocess predicted masks for metric computation.

        Args:
            mask: Probability mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the number
                of object instances in the prediction.

        Returns:
            Binary tensor of shape (B, P, K), where P is the number of points. If `use_threshold` is
            True, overlapping objects for the same point are possible.
        """
        if mask.ndim == 5:  # Video case
            mask = mask.flatten(0, 1)
        mask = mask.flatten(-2, -1)

        if self.use_threshold:
            mask = mask > self.threshold
            mask = mask.transpose(1, 2)
        else:
            maximum, indices = torch.max(mask, dim=1)
            mask = torch.nn.functional.one_hot(indices, num_classes=mask.shape[1])
            mask[:, :, 0][maximum == 0.0] = 0

        return mask

    def preprocess_ground_truth_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Preprocess ground truth mask for metric computation.

        Args:
            mask: Mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the number of object
                instances in the image. Class ID of objects is encoded as the value, i.e. densely
                represented.

        Returns:
            One-hot tensor of shape (B, P, J), where J is the number of the classes and P the number
            of points, with object instances with the same class ID merged together. In the case of
            an overlap of classes for a point, the class with the highest ID is assigned to that
            point.
        """
        if mask.ndim == 5:  # Video case
            mask = mask.flatten(0, 1)
        mask = mask.flatten(-2, -1)

        # Pixels which contain no object get assigned the background class 0. This also handles the
        # padding of zero masks which is done in preprocessing for batching.
        mask = torch.nn.functional.one_hot(
            mask.max(dim=1).values.to(torch.long), num_classes=self.n_classes_with_bg
        )

        return mask

    def compute(self):
        """Compute per-class IoU using matching."""
        if self.ignore_background:
            n_classes = self.n_classes
            confusion_mat = self.confusion_mat[:, 1:]
        else:
            n_classes = self.n_classes_with_bg
            confusion_mat = self.confusion_mat

        pairwise_iou, _, _, area_gt = self._compute_iou_from_confusion_mat(confusion_mat)

        if self.use_unmatched_as_background:
            # Match only in foreground
            pairwise_iou = pairwise_iou[1:, 1:]
            confusion_mat = confusion_mat[1:, 1:]
        else:
            # Predicted class zero is not matched against anything
            pairwise_iou = pairwise_iou[1:]
            confusion_mat = confusion_mat[1:]

        if self.matching == "hungarian":
            cluster_idxs, class_idxs = scipy.optimize.linear_sum_assignment(
                pairwise_iou.cpu(), maximize=True
            )
            cluster_idxs = torch.as_tensor(
                cluster_idxs, dtype=torch.int64, device=self.confusion_mat.device
            )
            class_idxs = torch.as_tensor(
                class_idxs, dtype=torch.int64, device=self.confusion_mat.device
            )
            matched_iou = pairwise_iou[cluster_idxs, class_idxs]
            true_pos = confusion_mat[cluster_idxs, class_idxs]

            if self.use_unmatched_as_background:
                cluster_oh = torch.nn.functional.one_hot(
                    cluster_idxs, num_classes=pairwise_iou.shape[0]
                )
                matched_clusters = cluster_oh.max(dim=0).values.to(torch.bool)
                bg_pred = self.confusion_mat[:1]
                bg_pred += self.confusion_mat[1:][~matched_clusters].sum(dim=0)
                bg_iou, _, _, _ = self._compute_iou_from_confusion_mat(bg_pred, area_gt)
                class_idxs = torch.cat((torch.zeros_like(class_idxs[:1]), class_idxs + 1))
                matched_iou = torch.cat((bg_iou[0, :1], matched_iou))
                true_pos = torch.cat((bg_pred[0, :1], true_pos))

        elif self.matching == "majority":
            max_iou, class_idxs = torch.max(pairwise_iou, dim=1)
            # Form new clusters by merging old clusters which are assigned the same ground truth
            # class. After merging, the number of clusters equals the number of classes.
            _, old_to_new_cluster_idx = torch.unique(class_idxs, return_inverse=True)

            confusion_mat_new = torch.zeros(
                n_classes, n_classes, dtype=torch.int64, device=self.confusion_mat.device
            )
            for old_cluster_idx, new_cluster_idx in enumerate(old_to_new_cluster_idx):
                if max_iou[old_cluster_idx] > 0.0:
                    confusion_mat_new[new_cluster_idx] += confusion_mat[old_cluster_idx]

            # Important: use previously computed area_gt because it includes background predictions,
            # whereas the new confusion matrix does not contain the bg predicted class anymore.
            pairwise_iou, _, _, _ = self._compute_iou_from_confusion_mat(confusion_mat_new, area_gt)
            max_iou, class_idxs = torch.max(pairwise_iou, dim=1)
            valid = max_iou > 0.0  # Ignore clusters without any kind of overlap
            class_idxs = class_idxs[valid]
            cluster_idxs = torch.arange(pairwise_iou.shape[1])[valid]
            matched_iou = pairwise_iou[cluster_idxs, class_idxs]
            true_pos = confusion_mat_new[cluster_idxs, class_idxs]

        iou = torch.zeros(n_classes, dtype=torch.float64, device=pairwise_iou.device)
        iou[class_idxs] = matched_iou

        accuracy = true_pos.sum().to(torch.float64) / area_gt.sum()
        empty_classes = area_gt == 0

        return iou, accuracy, empty_classes

    @staticmethod
    def _compute_iou_from_confusion_mat(
        confusion_mat: torch.Tensor, area_gt: Optional[torch.Tensor] = None
    ):
        area_pred = torch.sum(confusion_mat, axis=1)
        if area_gt is None:
            area_gt = torch.sum(confusion_mat, axis=0)
        union = area_pred.unsqueeze(1) + area_gt.unsqueeze(0) - confusion_mat
        pairwise_iou = confusion_mat.to(torch.float64) / union

        # Ignore classes that occured on no image.
        pairwise_iou[union == 0] = 0.0

        return pairwise_iou, union, area_pred, area_gt


def _remap_one_hot_mask(
    mask: torch.Tensor, new_classes: torch.Tensor, n_new_classes: int, strip_empty: bool = False
):
    """Remap classes from binary mask to new classes.

    In the case of an overlap of classes for a point, the new class with the highest ID is
    assigned to that point. If no class is assigned to a point, the point will have no class
    assigned after remapping as well.

    Args:
        mask: Binary mask of shape (B, P, K) where K is the number of old classes and P is the
            number of points.
        new_classes: Tensor of shape (B, K) containing ids of new classes for each old class.
        n_new_classes: Number of classes after remapping, i.e. highest class id that can occur.
        strip_empty: Whether to remove the empty pixels mask

    Returns:
        Tensor of shape (B, P, J), where J is the new number of classes.
    """
    assert new_classes.shape[1] == mask.shape[2]
    mask_dense = (mask * new_classes.unsqueeze(1)).max(dim=-1).values
    mask = torch.nn.functional.one_hot(mask_dense.to(torch.long), num_classes=n_new_classes + 1)

    if strip_empty:
        mask = mask[..., 1:]

    return mask

import tqdm
def compute_unsupervised_metrics(predictions, ground_truth, output_folder):
    print('Computing the CorLoc and DecRate metric...')
    cor_loc_metric = UnsupervisedBboxIoUMetricMinimal(matching="best_overlap",
                                                      correct_localization=True)
    det_rate_metric = UnsupervisedBboxIoUMetricMinimal(matching="best_overlap",
                                                       compute_discovery_fraction=True)

    image_ids = list(sorted(predictions.keys()))
    for image_id in tqdm.tqdm(image_ids):
        prediction = predictions[image_id]
        gt_boxlist = ground_truth[image_id]
        cor_loc_metric.update(prediction.bbox.unsqueeze(0), gt_boxlist.bbox.unsqueeze(0))
        det_rate_metric.update(prediction.bbox.unsqueeze(0), gt_boxlist.bbox.unsqueeze(0))

    result_str = "\nCorLoc: {}.\t DecRate: {}".format(cor_loc_metric.compute(), det_rate_metric.compute())
    print(result_str)
    if output_folder:
        output_file = os.path.join(output_folder, "result.txt")
        assert os.path.isfile(output_file)
        with open(output_file, "a") as fid:
            fid.write(result_str)

