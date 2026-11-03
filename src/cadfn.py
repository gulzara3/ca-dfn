"""
The Confidence-Aware Dynamic Fusion Network (CA-DFN) and its
entropy-regularised training objective.

CA-DFN computes per-sample attention weights over N base models from three
feature groups:
  1. per-model confidence  [softmax p_i, entropy H(p_i), max p_i]
  2. architecture-type embeddings (CNN=0, ViT=1) -> R^16
  3. pairwise KL-divergence disagreement for all C(N,2) model pairs
A temperature-scaled softmax produces the attention weights; an entropy
regulariser on those weights prevents collapse onto a single model.
"""
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class CADFN(nn.Module):
    def __init__(self, n_models, n_classes, model_types, hidden=512,
                 drop=0.3, attn_temp=3.0, input_drop=0.25):
        super().__init__()
        self.n_models = n_models
        self.n_classes = n_classes
        self.temp = attn_temp
        self.input_drop = input_drop
        self.arch_types = model_types

        self.arch_embed = nn.Embedding(2, 16)  # CNN=0, Transformer=1

        per_model = n_classes + 1 + 1 + 16     # softmax + entropy + maxprob + arch_emb
        n_pairs = n_models * (n_models - 1) // 2
        inp_dim = n_models * per_model + n_pairs

        self.net = nn.Sequential(
            nn.Linear(inp_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop),
        )
        self.res_block = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(drop), nn.Linear(hidden, hidden),
        )
        self.res_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, n_models)

    def forward(self, prob_list, training=False):
        B = prob_list[0].size(0)
        device = prob_list[0].device

        # Input-model dropout: replace one model's probs with a uniform dist.
        if training and self.input_drop > 0 and random.random() < self.input_drop:
            di = random.randint(0, self.n_models - 1)
            prob_list = [p if i != di else torch.ones_like(p) / self.n_classes
                         for i, p in enumerate(prob_list)]

        stk = torch.stack(prob_list, dim=1)  # (B, N, K)

        # Per-model confidence features + architecture embeddings
        mf = []
        for i, p in enumerate(prob_list):
            ent = -(p * (p + 1e-8).log()).sum(1, keepdim=True)
            mx = p.max(1, keepdim=True).values
            ae = self.arch_embed(
                torch.tensor([self.arch_types[i]], device=device)
            ).expand(B, -1)
            mf.append(torch.cat([p, ent, mx, ae], dim=1))
        model_feat = torch.cat(mf, dim=1)

        # Pairwise KL-divergence disagreement
        dis = []
        for i in range(self.n_models):
            for j in range(i + 1, self.n_models):
                kl = F.kl_div((prob_list[i] + 1e-8).log(), prob_list[j] + 1e-8,
                              reduction='none').sum(1, keepdim=True)
                dis.append(kl)

        x = torch.cat([model_feat, torch.cat(dis, dim=1)], dim=1)
        h = self.net(x)
        h = self.res_norm(h + self.res_block(h))         # residual block
        w = F.softmax(self.head(h) / self.temp, dim=1)   # temperature-scaled attention
        fused = (w.unsqueeze(2) * stk).sum(1)            # weighted combination
        return fused, w


class EntropyRegLoss(nn.Module):
    """L = CE(p_hat, y) - lambda * H(w) / log(N)."""
    def __init__(self, ent_w=0.3):
        super().__init__()
        self.ent_w = ent_w

    def forward(self, ens_probs, labels, attn_w):
        ce = F.nll_loss(ens_probs.clamp(min=1e-8).log(), labels)
        ent = -(attn_w * (attn_w + 1e-8).log()).sum(1).mean()
        norm_ent = ent / math.log(attn_w.size(1))
        loss = ce - self.ent_w * norm_ent
        return loss, ce.item(), norm_ent.item()
