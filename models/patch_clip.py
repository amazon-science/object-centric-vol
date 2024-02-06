import clip
import math
import torch
from torch import nn
from collections import OrderedDict


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class VisionEmbedder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(VisionEmbedder, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(out_dim),
            nn.Linear(out_dim, out_dim)
        )
        self.res = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.res(x) + self.main(x)


class PatchCLIP(nn.Module):
    def __init__(self, model_name="ViT-B/16"):
        super(PatchCLIP, self).__init__()
        model, _ = clip.load(model_name, device='cpu')
        self.model = model.visual.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        hidden_dim, output_dim = model.visual.proj.shape
        num_heads = self.model.transformer.resblocks[-1].attn.num_heads

        self.last_transformer = ResidualAttentionBlock(hidden_dim, num_heads)
        self.ln_last = LayerNorm(hidden_dim)
        self.vision_emb = VisionEmbedder(hidden_dim, output_dim)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.model.positional_embedding.shape[-2] - 1

        patch_size = 16
        if npatch == N and w == h:
            return self.model.positional_embedding
        class_pos_embed = self.model.positional_embedding[0]
        patch_pos_embed = self.model.positional_embedding[1:]
        dim = patch_pos_embed.shape[-1]
        w0 = w // patch_size
        h0 = h // patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            size=(w0, h0),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0).unsqueeze(0), patch_pos_embed), dim=1)
    
    def forward(self, x):
        with torch.no_grad():
            H, W = x.shape[-2:]
            x = self.model.conv1(x)                     # shape = [*, hidden_dim, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)   # shape = [*, hidden_dim, grid**2]
            x = x.permute(0, 2, 1)                      # shape = [*, grid**2, hidden_dim]
            x = torch.cat([self.model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, 1 + grid**2, hidden_dim]
            # x = x + self.model.positional_embedding.to(x.dtype)
            x = x + self.interpolate_pos_encoding(x, H, W).to(x.dtype)

            # Attention in Backbone
            x = self.model.ln_pre(x)
            x = x.permute(1, 0, 2)  # NLD -> LND
            before_last_x = self.model.transformer.resblocks[:-1](x)
            x = self.model.transformer.resblocks[-1](before_last_x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            cls_token = self.model.ln_post(x[:, 0, :])

            if self.model.proj is not None:
                visual_feat = cls_token @ self.model.proj       # final visual features to be matched with text
                original_patch_feat = self.model.ln_post(x[:, 1:, :]) @ self.model.proj

        new_x = self.last_transformer(before_last_x.detach())
        new_x = new_x.permute(1, 0, 2)  # LND -> NLD
        new_x = self.ln_last(new_x[:, 1:, :])       # can exclude CLS token, only operate patch features
        patch_feat = self.vision_emb(new_x)

        query = visual_feat.detach()[:,None,:]      # use detach to double confirm there is no gradient
        key = patch_feat.permute(0, 2, 1)

        x_attn = (query @ key).softmax(-1)      # [N, 1, D] x [N, D, L] -> [N, 1, L]
        patch_pool = (x_attn @ patch_feat)[:, 0, :]     # [N, 1, L] x [N, L D] -> [N, 1, D] -> [N, D]

        visual_feat_normalized = visual_feat / visual_feat.norm(dim=-1, keepdim=True)
        patch_pool_normalized = patch_pool / patch_pool.norm(dim=-1, keepdim=True)
        loss_mat = (patch_pool_normalized @ visual_feat_normalized.detach().T).exp()
        loss_diag = loss_mat.diag()
        loss_denom = loss_mat.sum(1)
        loss_InfoNCE = -(loss_diag / loss_denom).log().mean()

        return loss_InfoNCE, visual_feat, patch_pool, x_attn, patch_feat, original_patch_feat

    def average_patch_video(self, x, mask_one_hot_idx, text_features, temperature=100, patch_feat=None):
        x = x.permute(0, 3, 1, 2)

        if patch_feat is None:
            with torch.no_grad():
                loss_InfoNCE, visual_feat, patch_pool, x_attn, patch_feat, original_patch_feat = self(x)
                patch_feat = patch_feat.flatten(0, 1)
        key = mask_one_hot_idx @ patch_feat
        key = key / (key.norm(dim=1, keepdim=True) + 1e-6)
        logits = temperature * key @ text_features.T

        return logits
