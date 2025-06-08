import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict

class AttentionFusion(nn.Module):
    def __init__(self, input_dims, fused_dim=512):
        super().__init__()
        # One Linear(d_i → fused_dim) per branch
        self.proj = nn.ModuleList([nn.Linear(d, fused_dim) for d in input_dims])

        # Attention MLP: fused_dim → hidden → 1 (per branch) → softmax
        attn_hidden = max(fused_dim // 4, 128)
        self.attn_score = nn.Sequential(
            nn.Linear(fused_dim, attn_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(attn_hidden, 1),
        )

        # Keep track of channel‐size to detect channels-last
        self.input_dims = input_dims

    def forward(self, feats):
        """
        feats: list of length N_branches.
               Each f_i is either:
                 - 2D tensor [B, D_i], or
                 - 4D tensor [B, C_i, H_i, W_i] (channels-first), or
                 - 4D tensor [B, H_i, W_i, C_i] (channels-last).

        Returns:
          fused:        [B, fused_dim]
          attn_logits:  [B, N_branches]
          projected:    list of length N_branches, each [B, fused_dim]
        """
        projected = []
        B = None

        for i, f_i in enumerate(feats):
            # If f_i is 4D, detect channels-first vs. channels-last, then pool to [B, C]
            if f_i.dim() == 4:
                # Check channel location
                if f_i.shape[1] == self.input_dims[i]:
                    tensor_cf = f_i                          # [B, C, H, W]
                elif f_i.shape[-1] == self.input_dims[i]:
                    tensor_cf = f_i.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
                else:
                    raise RuntimeError(
                        f"Branch {i}: expected channel‐size={self.input_dims[i]} "
                        f"at dim=1 or dim=3, but got {f_i.shape}"
                    )
                pooled = F.adaptive_avg_pool2d(tensor_cf, 1)  # [B, C, 1, 1]
                x = pooled.view(pooled.size(0), -1)           # [B, C]
            elif f_i.dim() == 2:
                x = f_i  # already [B, D_i]
            else:
                raise RuntimeError(f"Unexpected feat dims {f_i.shape}")

            if B is None:
                B = x.size(0)

            # Project into [B, fused_dim]
            p_i = self.proj[i](x)  # [B, fused_dim]
            projected.append(p_i)

        # Stack projected: [B, N, fused_dim]
        stacked = torch.stack(projected, dim=1)
        B, N, D = stacked.shape

        # Compute attention scores per projected vector
        flat = stacked.view(B * N, D)                    # [B*N, D]
        scores = self.attn_score(flat).view(B, N)        # [B, N]
        weights = F.softmax(scores, dim=1)               # [B, N]

        # Weighted sum → fused [B, fused_dim]
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # [B, fused_dim]

        return fused, weights, projected


class Judge(nn.Module):
    def __init__(
        self,
        branches: Dict[str, nn.Module],
        fused_dim: int = 512,
        fusion_level: str = "feature",
    ):
        super().__init__()
        assert fusion_level in ("feature", "decision")
        self.branch_names = list(branches.keys())
        self.branches = nn.ModuleDict(branches)
        self.fusion_level = fusion_level
        self.num_branches = len(self.branch_names)

        if fusion_level == "feature":
            input_dims = [2048, 256, 768, 2048, 64]
            assert len(input_dims) == self.num_branches
            self.input_dims = input_dims
            self.fuser = AttentionFusion(input_dims, fused_dim=fused_dim)
            self.classifier = nn.Linear(fused_dim, 2)
        else:
            # Add a head for the image‐branch only: 64 → 2
            self.img_branch_head = nn.Linear(64, 2)
            self.input_dims = None
            self.decision_classifier = nn.Sequential(
                nn.Linear(self.num_branches, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(32, 2),
            )

    def forward(self, img1=None, img2=None, img3=None, img4=None, video=None, labels=None):
        if self.fusion_level == "feature":
            feats = []

            # 1) SPSL branch
            out0 = self.branches["spsl"]({"image": img1, "label": labels})
            r0 = out0["feat"]  # maybe [B, 2048] or 4D
            if r0.dim() == 4:
                # detect channels-last
                if r0.shape[-1] == self.input_dims[0]:
                    r0 = r0.permute(0, 3, 1, 2).contiguous()
                r0 = F.adaptive_avg_pool2d(r0, 1).view(r0.size(0), -1)  # → [B, 2048]
            feats.append(r0)

            # 2) UCF branch
            out1 = self.branches["ucf"]({"image": img2,"label": labels},inference=True)
            r1 = out1["feat"]  # maybe [B, 256] or 4D
            if r1.dim() == 4:
                if r1.shape[-1] == self.input_dims[1]:
                    r1 = r1.permute(0, 3, 1, 2).contiguous()
                r1 = F.adaptive_avg_pool2d(r1, 1).view(r1.size(0), -1)  # → [B, 256]
            feats.append(r1)

            # 3) UIA‐ViT branch
            out2 = self.branches["uiavit"]({"image": img3, "label": labels})
            r2 = out2["feat"]  # maybe [B, 14, 14, 768] or [B, 768, 14, 14]
            if r2.dim() == 4:
                if r2.shape[-1] == self.input_dims[2]:
                    r2 = r2.permute(0, 3, 1, 2).contiguous()
                r2 = F.adaptive_avg_pool2d(r2, 1).view(r2.size(0), -1)  # → [B, 768]
            feats.append(r2)

            # 4) STIL branch
            out3 = self.branches["stil"]({"image": video, "label": labels})
            r3 = out3["feat"]  # maybe [B, 2048, 7, 7]
            if r3.dim() == 4:
                if r3.shape[-1] == self.input_dims[3]:
                    r3 = r3.permute(0, 3, 1, 2).contiguous()
                r3 = F.adaptive_avg_pool2d(r3, 1).view(r3.size(0), -1)  # → [B, 2048]
            feats.append(r3)

            # 5) raw-image branch
            if "img_branch" in self.branches:
                r4 = self.branches["img_branch"](img4)  # [B, 64]
                if r4.dim() == 4:
                    if r4.shape[-1] == self.input_dims[4]:
                        r4 = r4.permute(0, 3, 1, 2).contiguous()
                    r4 = F.adaptive_avg_pool2d(r4, 1).view(r4.size(0), -1)  # → [B, 64]
                feats.append(r4)

            #`feats` is a list of 5 tensors, each 2D: [B, D_i].
            #fuser now returns (fused, attn_weights, projected_list).
            super_r, attn_weights, projected_list = self.fuser(feats)
            logits = self.classifier(super_r)  # [B, 2]

            # the projected_list (for alignment), and the fused vector.
            return {
                "logits":        logits,          # [B, 2]
                "branch_feats":  feats,           # list of [B, D_i]
                "projected_feats": projected_list,# list of [B, fused_dim]
                "fused_feat":    super_r,         # [B, fused_dim]
                "attn":          attn_weights,    # [B, N_branches]
            }

        else:
            # ── Decision‐level fusion ──
            branch_logits = []

            out0 = self.branches["spsl"]({"image": img1, "label": labels})
            l0 = out0["cls"][:, 1].unsqueeze(1)  # [B, 1]
            branch_logits.append(l0)

            out1 = self.branches["ucf"]({"image": img2, "label": labels}, inference= True)
            l1 = out1["cls"][:, 1].unsqueeze(1)
            branch_logits.append(l1)

            out2 = self.branches["uiavit"]({"image": img3, "label": labels})
            l2 = out2["cls"][:, 1].unsqueeze(1)
            branch_logits.append(l2)

            out3 = self.branches["stil"]({"image": video, "label": labels})
            l3 = out3["cls"][:, 1].unsqueeze(1)
            branch_logits.append(l3)

            if "img_branch" in self.branches:
                out4 = self.branches["img_branch"](img4)
                logits4 = self.img_branch_head(out4)
                l4 = logits4[:, 1].unsqueeze(1)
                branch_logits.append(l4)

            logits_tensor = torch.cat(branch_logits, dim=1)  # [B, N_branches]
            logits = self.decision_classifier(logits_tensor) # [B, 2]

            return {
                "logits":        logits,
                "branch_logits": branch_logits,  # each [B, 1]
            }

class JudgeLoss(nn.Module):
    def __init__(self, fusion_mode="feature", λ: float = 0.1, μ: float = 0.1):
        super().__init__()
        assert fusion_mode in ("feature", "decision")
        self.fusion_mode = fusion_mode
        self.λ = λ
        self.μ = μ

    @staticmethod
    def balance_loss(reps):
        """
        reps: list of tensors, each shape (B, D_i)
        → norms: list of (B,) → stack → (B, branches)
        → return variance over branches per sample, then mean over batch
        """
        norms = torch.stack([r.norm(dim=1) for r in reps], dim=1)  # (B, branches)
        return norms.var(dim=1).mean()

    @staticmethod
    def alignment_loss(projected_reps, super_r):
        """
        projected_reps: list of (B, D_fused)  (all in fused_dim space)
        super_r:        (B, D_fused)
        We compute cosine_similarity on each projected_i vs. super_r.
        """
        loss = 0.0
        for p_i in projected_reps:
            cos = F.cosine_similarity(p_i, super_r, dim=1)  # (B,)
            loss += (1.0 - cos).mean()
        return loss / len(projected_reps)

    def forward(self, output: Dict[str, torch.Tensor], labels: torch.Tensor):
        logits = output["logits"]  # (B, num_classes)

        # Ensure labels is 1-D LongTensor of shape [B]:
        if labels.dim() == 2 and labels.size(1) == logits.size(1):
            # one-hot → indices
            labels = labels.argmax(dim=1)
        else:
            labels = labels.view(-1).long()

        if self.fusion_mode == "decision":
            loss_cls = F.cross_entropy(logits, labels)
            return loss_cls, {"loss_cls": loss_cls.item()}

        # feature‐fusion mode:
        # 1) classification loss
        loss_cls = F.cross_entropy(logits, labels)

        # 2) balance‐loss on the *projected* features + fused
        projected = output["projected_feats"]  # each [B, fused_dim]
        fused_feat = output["fused_feat"]      # [B, fused_dim]
        l_bal = JudgeLoss.balance_loss(projected + [fused_feat])

        # 3) alignment‐loss on the *projected* features vs fused
        l_aln = JudgeLoss.alignment_loss(projected, fused_feat)

        total_loss = loss_cls + self.λ * l_bal + self.μ * l_aln
        return total_loss, {
            "loss_cls":     loss_cls.item(),
            "loss_balance": l_bal.item(),
            "loss_align":   l_aln.item(),
            "loss_total":   total_loss.item(),
        }
