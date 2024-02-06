import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from typing import Callable, Optional, List, Union, Tuple
from torchvision import transforms
from timm.models import create_model
from .videomae import VisionTransformer

class RandomConditioning(nn.Module):
    """Random conditioning with potentially learnt mean and stddev."""

    def __init__(
        self,
        object_dim: int,
        n_slots: int,
        learn_mean: bool = True,
        learn_std: bool = True,
        mean_init: Optional[Callable[[torch.Tensor], None]] = None,
        logsigma_init: Optional[Callable[[torch.Tensor], None]] = None
    ):
        super().__init__()
        self.n_slots = n_slots
        self.object_dim = object_dim

        if learn_mean:
            self.slots_mu = nn.Parameter(torch.zeros(1, 1, object_dim))
        else:
            self.register_buffer("slots_mu", torch.zeros(1, 1, object_dim))

        if learn_std:
            self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, object_dim))
        else:
            self.register_buffer("slots_logsigma", torch.zeros(1, 1, object_dim))

        if mean_init is None:
            mean_init = nn.init.xavier_uniform_
        if logsigma_init is None:
            logsigma_init = nn.init.xavier_uniform_

        with torch.no_grad():
            mean_init(self.slots_mu)
            logsigma_init(self.slots_logsigma)

    def forward(self, batch_size: int):
        mu = self.slots_mu.expand(batch_size, self.n_slots, -1)
        sigma = self.slots_logsigma.exp().expand(batch_size, self.n_slots, -1)
        return mu + sigma * torch.randn_like(mu)


def get_activation_fn(name: str, inplace: bool = True, leaky_relu_slope: Optional[float] = None):
    if callable(name):
        return name

    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    elif name == "leaky_relu":
        if leaky_relu_slope is None:
            raise ValueError("Slope of leaky ReLU was not defined")
        return nn.LeakyReLU(leaky_relu_slope, inplace=inplace)
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "identity":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation function {name}")


class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return inputs + self.module(inputs)


def build_mlp(
    input_dim: int,
    output_dim: int,
    features: List[int],
    activation_fn: Union[str, Callable] = "relu",
    final_activation_fn: Optional[Union[str, Callable]] = None,
    initial_layer_norm: bool = False,
    residual: bool = False,
) -> nn.Sequential:
    layers = []
    current_dim = input_dim
    if initial_layer_norm:
        layers.append(nn.LayerNorm(current_dim))

    for n_features in features:
        layers.append(nn.Linear(current_dim, n_features))
        nn.init.zeros_(layers[-1].bias)
        layers.append(get_activation_fn(activation_fn))
        current_dim = n_features

    layers.append(nn.Linear(current_dim, output_dim))
    nn.init.zeros_(layers[-1].bias)
    if final_activation_fn is not None:
        layers.append(get_activation_fn(final_activation_fn))

    if residual:
        return Residual(nn.Sequential(*layers))
    return nn.Sequential(*layers)


def build_two_layer_mlp(
    input_dim, output_dim, hidden_dim, initial_layer_norm: bool = False, residual: bool = False
):
    """Build a two layer MLP, with optional initial layer norm.

    Separate class as this type of construction is used very often for slot attention and
    transformers.
    """
    return build_mlp(
        input_dim, output_dim, [hidden_dim], initial_layer_norm=initial_layer_norm, residual=residual
    )


class SlotAttention(nn.Module):
    """Implementation of SlotAttention.

    Based on the slot attention implementation of Phil Wang available at:
    https://github.com/lucidrains/slot-attention
    """

    def __init__(
        self,
        dim: int,
        feature_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.iters = iters
        self.eps = eps
        self.use_implicit_differentiation = use_implicit_differentiation

        if kvq_dim is None:
            self.kvq_dim = dim
        else:
            self.kvq_dim = kvq_dim

        if self.kvq_dim % self.n_heads != 0:
            raise ValueError("Key, value, query dimensions must be divisible by number of heads.")
        self.dims_per_head = self.kvq_dim // self.n_heads
        self.scale = self.dims_per_head**-0.5

        self.to_q = nn.Linear(dim, self.kvq_dim, bias=use_projection_bias)
        self.to_k = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)
        self.to_v = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)

        self.gru = nn.GRUCell(self.kvq_dim, dim)

        self.norm_input = nn.LayerNorm(feature_dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.ff_mlp = ff_mlp

    def step(self, slots, k, v, masks=None):
        bs, n_slots, _ = slots.shape
        slots_prev = slots

        slots = self.norm_slots(slots)
        q = self.to_q(slots).view(bs, n_slots, self.n_heads, self.dims_per_head)

        dots = torch.einsum("bihd,bjhd->bihj", q, k) * self.scale
        if masks is not None:
            # Masked slots should not take part in the competition for features. By replacing their
            # dot-products with -inf, their attention values will become zero within the softmax.
            dots.masked_fill_(masks.to(torch.bool).view(bs, n_slots, 1, 1), float("-inf"))

        attn = dots.flatten(1, 2).softmax(dim=1)  # Take softmax over slots and heads
        attn = attn.view(bs, n_slots, self.n_heads, -1)
        attn_before_reweighting = attn
        attn = attn + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum("bjhd,bihj->bihd", v, attn)

        slots = self.gru(updates.reshape(-1, self.kvq_dim), slots_prev.reshape(-1, self.dim))

        slots = slots.reshape(bs, -1, self.dim)

        if self.ff_mlp:
            slots = self.ff_mlp(slots)

        return slots, attn_before_reweighting.mean(dim=2)

    def iterate(self, slots, k, v, masks=None):
        for _ in range(self.iters):
            slots, attn = self.step(slots, k, v, masks)
        return slots, attn

    def forward(
        self, inputs: torch.Tensor, conditioning: torch.Tensor, masks: Optional[torch.Tensor] = None
    ):
        b, n, d = inputs.shape
        slots = conditioning

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs).view(b, n, self.n_heads, self.dims_per_head)
        v = self.to_v(inputs).view(b, n, self.n_heads, self.dims_per_head)

        if self.use_implicit_differentiation:
            slots, attn = self.iterate(slots, k, v, masks)
            slots, attn = self.step(slots.detach(), k, v, masks)
        else:
            slots, attn = self.iterate(slots, k, v, masks)

        return slots, attn


class SlotAttentionGrouping(nn.Module):
    """Implementation of SlotAttention for perceptual grouping.

    Args:
        feature_dim: Dimensionality of features to slot attention (after positional encoding).
        object_dim: Dimensionality of slots.
        kvq_dim: Dimensionality after projecting to keys, values, and queries. If `None`,
            `object_dim` is used.
        n_heads: Number of heads slot attention uses.
        iters: Number of slot attention iterations.
        eps: Epsilon in slot attention.
        ff_mlp: Optional module applied slot-wise after GRU update.
        positional_embedding: Optional module applied to the features before slot attention, adding
            positional encoding.
        use_projection_bias: Whether to use biases in key, value, query projections.
        use_implicit_differentiation: Whether to use implicit differentiation trick. If true,
            performs one more iteration of slot attention that is used for the gradient step after
            `iters` iterations of slot attention without gradients. Faster and more memory efficient
            than the standard version, but can not backpropagate gradients to the conditioning input.
        input_dim: Dimensionality of features before positional encoding is applied. Specifying this
            is optional but can be convenient to structure configurations.
    """

    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        use_empty_slot_for_masked_slots: bool = False,
        slot_mask_path: Optional[str] = None,
    ):
        super().__init__()

        self._object_dim = object_dim
        self.slot_attention = SlotAttention(
            dim=object_dim,
            feature_dim=object_dim,
            kvq_dim=kvq_dim,
            n_heads=n_heads,
            iters=iters,
            eps=eps,
            ff_mlp=ff_mlp,
            use_projection_bias=use_projection_bias,
            use_implicit_differentiation=use_implicit_differentiation,
        )

        self.positional_embedding = build_two_layer_mlp(input_dim=feature_dim,
                                                        output_dim=object_dim,
                                                        hidden_dim=feature_dim,
                                                        initial_layer_norm=True)

        if use_empty_slot_for_masked_slots:
            if slot_mask_path is None:
                raise ValueError("Need `slot_mask_path` for `use_empty_slot_for_masked_slots`")
            self.empty_slot = nn.Parameter(torch.randn(object_dim) * object_dim**-0.5)
        else:
            self.empty_slot = None

    def forward(
        self,
        extracted_features,
        conditioning,
        slot_masks=None,
    ):
        features = self.positional_embedding(extracted_features)
        slots, attn = self.slot_attention(features, conditioning, slot_masks)
        if slot_masks is not None and self.empty_slot is not None:
            slots[slot_masks] = self.empty_slot.to(dtype=slots.dtype)

        return slots, attn

class PatchDecoderVideo(nn.Module):
    def __init__(
        self,
        object_dim: int,
        output_dim: int,
        num_patches_per_frame: int,
        num_frames: int,
        decoder: nn.Module,
        decoder_input_dim: Optional[int] = None,
        resize: int = 256
    ):
        nn.Module.__init__(self)
        self.output_dim = output_dim
        self.num_patches_per_frame = num_patches_per_frame
        self.num_frames = num_frames
        self.resize = resize

        if decoder_input_dim is not None:
            self.inp_transform = nn.Linear(object_dim, decoder_input_dim, bias=True)
            nn.init.xavier_uniform_(self.inp_transform.weight)
            nn.init.zeros_(self.inp_transform.bias)
        else:
            self.inp_transform = None
            decoder_input_dim = object_dim

        self.decoder = decoder
        self.pos_embed = nn.Parameter(torch.randn(self.num_frames, self.num_patches_per_frame, decoder_input_dim) * 0.02)

    def forward(self, object_features: torch.Tensor, nh, nw):
        assert object_features.dim() >= 3   # Image or video data.  (b, s, d)

        initial_shape = object_features.shape[:-1]              # (b, s)
        object_features = object_features.flatten(0, -2)        # (b*s, d)

        if self.inp_transform is not None:
            object_features = self.inp_transform(object_features)

        # duplicate the slot representation into each patch, (b*s, t*n, d)
        object_features = object_features.unsqueeze(1).expand(-1, self.num_frames*nw*nh, -1)

        # Simple learned additive embedding as in ViT
        N = self.num_patches_per_frame
        dim = self.pos_embed.shape[-1]
        patch_pos_embed = nn.functional.interpolate(
            self.pos_embed.reshape(self.num_frames, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            size=(nh, nw),
            mode='bicubic',
            align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        object_features = object_features + patch_pos_embed

        output = self.decoder(object_features)              # (b*s, t*n, d+1)
        output = output.unflatten(0, initial_shape)         # (b, s, t*n, d+1)

        # Split out alpha channel and normalize over slots.
        decoded_patches, alpha = output.split([self.output_dim, 1], dim=-1)     # (b, s, t*n, d), (b, s, t*n, 1)
        alpha = alpha.softmax(dim=-3)       # (b, s, t*n, 1)

        reconstruction = torch.sum(decoded_patches * alpha, dim=-3)     # (b, t*n, d)
        masks = alpha.squeeze(-1)           # (b, s, t*n)

        masks = rearrange(masks, "b s (t n) -> (b t) s n", t=self.num_frames)       # (b*t, s n)

        masks_as_image = resize_patches_to_image_non_square(
            masks,
            size=(nh, nw),
            resize_mode="bilinear"
        )

        return reconstruction, masks, masks_as_image

def resize_patches_to_image_non_square(patches, size, resize_mode):
    H, W = size
    n_channels = patches.shape[-2]
    image = torch.nn.functional.interpolate(
        patches.view(-1, n_channels, H, W),
        scale_factor=(16.0, 16.0),
        mode=resize_mode,
    )
    return image.view(*patches.shape[:-1], image.shape[-2], image.shape[-1])

class GroupingVideoMAE(nn.Module):
    """
    Spatiotemporal Grouping for non-square multi-resolution input video.
    """
    def __init__(
        self,
        checkpoint_path,
        object_dim=128,
        n_slots=24,
        feat_dim=768,
        num_patches=256,
        num_frames=4,
        img_size=224
    ):
        super().__init__()
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.img_size = img_size

        # conditioning
        self.conditioning = RandomConditioning(object_dim=object_dim, n_slots=n_slots)

        # feature extractor
        self.model = create_model(
            "videomae_vit_base_patch16_224",
            pretrained=False,
            num_classes=174,
            all_frames=16 * 1,
            drop_rate=0.0,
            drop_path_rate=0.1,
            attn_drop_rate=0.0,
            drop_block_rate=None,
            use_mean_pooling=False,
            init_scale=0.001,
        )

        ckpt = torch.load(checkpoint_path,
                          map_location='cpu')
        new_ckpt = {}
        for k, v in ckpt['model'].items():
            if 'encoder.' in k:
                new_ckpt[k.replace('encoder.', '')] = v
        self.model.load_state_dict(new_ckpt, strict=False)

        self.model.requires_grad_(False)
        self.model.eval()

        # perceptual grouping
        ff_mlp = build_two_layer_mlp(input_dim=object_dim, output_dim=object_dim, hidden_dim=object_dim*4,
                                     initial_layer_norm=True, residual=True)
        self.grouping = SlotAttentionGrouping(feature_dim=feat_dim,
                                              object_dim=object_dim,
                                              use_projection_bias=False,
                                              ff_mlp=ff_mlp)

        # object decoder
        dec_mlp = build_mlp(input_dim=object_dim, output_dim=feat_dim+1, features=[1024, 1024, 1024])
        self.decoder = PatchDecoderVideo(object_dim=object_dim,
                                                 output_dim=feat_dim,
                                                 num_patches_per_frame=num_patches,
                                                 num_frames=num_frames,
                                                 decoder=dec_mlp,
                                                 resize=img_size)

    def forward(self, images):
        _, _, H, W, _ = images.shape            # (b, t, h, w, c)
        assert self.num_frames == images.shape[1]

        conditioning = self.conditioning(images.shape[0])     # (b, s, d)

        images = rearrange(images, 'b t h w c -> b c t h w')
        self.model.eval()
        features = self.model(images)  # (b, t*n, d)

        slots, attn = self.grouping(features, conditioning)         # (b, s, d), (b, s, t*n)

        reconstruction, masks, masks_as_image = self.decoder(slots, H//16, W//16)
        # (b, t*n, d),  (b, s, t*n),  (b*t, s, img_size, img_size)

        masks_as_image = rearrange(masks_as_image, '(n t) s h w -> n t s h w', t=self.num_frames)

        feat_recon_loss = F.mse_loss(reconstruction, features)

        return feat_recon_loss, masks_as_image

    def inference(self, images):
        _, _, H, W, _ = images.shape
        assert self.num_frames == images.shape[1]           # (b, t, h, w, c)
        conditioning = self.conditioning(images.shape[0])     # (b, s, d)
        images = rearrange(images, 'b t h w c -> b c t h w')
        self.model.eval()
        features = self.model(images)  # (b, t*n ,d)
        slots, attn = self.grouping(features, conditioning)         # (b, s, d), (b, s, t*n)
        reconstruction, masks, masks_as_image = self.decoder(slots, H//16, W//16)
        # (b, t*n, d),  (b*t, s, n),  (b, t, s, img_size, img_size)
        masks = rearrange(masks, '(n t) s l -> n t s l', t=self.num_frames)
        masks_as_image = rearrange(masks_as_image, '(n t) s h w -> n t s h w', t=self.num_frames)

        return masks, masks_as_image