"""Adapter wrapping NVIDIA's official pretrained MambaVision backbone
(NVlabs/MambaVision, CVPR 2025) so it exposes the same interface as
ResNet12Encoder / VimBackbone / SMNetConv64FEncoder.

Requires: pip install mambavision timm==1.0.15 transformers

NOTE ON LICENSE: MambaVision code is released under the NVIDIA Source Code
License-NC and pretrained weights under CC-BY-NC-SA-4.0 -- both
non-commercial only. Confirm your use case stays research/non-commercial
before deploying anything built on top of this encoder.

NOTE ON OUTPUT FORMAT (confirmed via official NVlabs/MambaVision README):
`AutoModel.from_pretrained(..., trust_remote_code=True)(inputs)` returns a
2-tuple `(out_avg_pool, features)` where `features` is a LIST of 4 stage
feature maps (not a single tensor). We take `features[-1]` (the deepest,
highest-channel stage) as the (B, C, H, W) map fed into PECT.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaVisionNvidiaEncoder(nn.Module):
    """Pretrained NVIDIA MambaVision backbone, adapted to the
    forward_features / out_channels / out_dim / feat_dim interface used
    throughout tim_2026.model._reference.encoders.
    """

    def __init__(
        self,
        image_size: int = 224,
        model_name: str = "nvidia/MambaVision-T-1K",
        pool_output: bool = False,
        freeze_early_stages: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "MambaVisionNvidiaEncoder requires the 'transformers' and "
                "'mambavision' packages. Install with:\n"
                "  pip install mambavision timm==1.0.15 transformers"
            ) from exc

        self.pool_output = bool(pool_output)
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # Dry run to discover the real output shape for this checkpoint/version
        # instead of hardcoding channel counts that differ across MambaVision
        # variants (T/T2/S/B/L/L2).
        self.model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            probe_out = self.model(dummy)
            feature_map = self._extract_feature_map(probe_out)

        self.out_channels = int(feature_map.shape[1])
        self.out_dim = self.out_channels
        self.out_spatial = int(feature_map.shape[-1])
        self.feat_dim = [self.out_channels, self.out_spatial, self.out_spatial]

        if freeze_early_stages:
            self._freeze_early_stages()

    def _extract_feature_map(self, out) -> torch.Tensor:
        """Normalize the HF AutoModel output into a (B, C, H, W) tensor.

        Confirmed real output shape (NVlabs/MambaVision README): a 2-tuple
        `(out_avg_pool, features)` where `features` is a list of 4 stage
        feature maps. We take the last (deepest) stage. Falls back to a
        few other shapes defensively for forward-compatibility with future
        transformers/mambavision releases.
        """
        if (
            isinstance(out, (tuple, list))
            and len(out) == 2
            and isinstance(out[1], (tuple, list))
        ):
            # (out_avg_pool, features) -- the documented MambaVision format.
            candidate = out[1][-1]
        elif isinstance(out, (tuple, list)):
            candidate = out[-1]
        elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            candidate = out.last_hidden_state
        elif hasattr(out, "hidden_states") and out.hidden_states is not None:
            candidate = out.hidden_states[-1]
        elif torch.is_tensor(out):
            candidate = out
        else:
            raise RuntimeError(
                "Khong nhan dien duoc dinh dang output cua MambaVision AutoModel "
                f"(kieu tra ve: {type(out)!r}). Chay debug: "
                "`out = model(dummy); print(type(out)); "
                "print(out.__dict__ if hasattr(out, '__dict__') else out)` "
                "trong Colab de xac dinh dung attribute/key can lay, roi sua "
                "lai ham _extract_feature_map cho khop."
            )

        if not torch.is_tensor(candidate):
            raise RuntimeError(
                f"Sau khi trich xuat, candidate van khong phai tensor "
                f"(kieu: {type(candidate)!r}). Cau truc output thuc te khac "
                "voi dinh dang da xac nhan tu README -- kiem tra lai."
            )

        if candidate.dim() == 3:
            batch, num_tokens, channels = candidate.shape
            side = int(num_tokens ** 0.5)
            if side * side != num_tokens:
                raise RuntimeError(
                    f"Output dang chuoi token (B={batch}, N={num_tokens}, "
                    f"C={channels}) nhung N khong phai so chinh phuong -- "
                    "khong the tu dong reshape ve luoi khong gian vuong. "
                    "Kiem tra lai xem co token [CLS] can bo di truoc khong."
                )
            candidate = candidate.transpose(1, 2).reshape(batch, channels, side, side)
        elif candidate.dim() != 4:
            raise RuntimeError(
                f"Feature map co so chieu khong mong doi: {tuple(candidate.shape)}. "
                "Can kiem tra lai cau truc output cua AutoModel."
            )
        return candidate

    def _freeze_early_stages(self) -> None:
        """Freeze the whole pretrained backbone except the last hierarchical
        stage. Recommended default when fine-tuning on a small dataset (PD
        scalograms, ~1.4k images) to avoid destroying the ImageNet features.
        """
        for param in self.model.parameters():
            param.requires_grad_(False)

        last_stage = None
        for attr_name in ("levels", "stages", "layers", "blocks"):
            container = getattr(self.model, attr_name, None)
            if container is not None and hasattr(container, "__len__") and len(container) > 0:
                last_stage = container[-1]
                break

        if last_stage is None:
            import warnings

            warnings.warn(
                "MambaVisionNvidiaEncoder: khong tim thay stage cuoi cung de "
                "mo dong bang (thu cac ten attribute levels/stages/layers/"
                "blocks deu khong co). Toan bo backbone dang bi dong bang -- "
                "chi head/adapter cua PECT se duoc train. Kiem tra lai ten "
                "attribute thuc te bang: print(list(dict(self.model.named_"
                "children()).keys())) roi sua _freeze_early_stages cho khop."
            )
            return

        for param in last_stage.parameters():
            param.requires_grad_(True)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return self._extract_feature_map(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.forward_features(x)
        if not self.pool_output:
            return feature_map
        return F.adaptive_avg_pool2d(feature_map, 1).view(feature_map.size(0), -1)
