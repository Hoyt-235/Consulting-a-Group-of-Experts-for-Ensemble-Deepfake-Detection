import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter



#L_total = L_cls + λ * L_balance + μ * L_alignment

def balance_loss(representations):
    norms = [torch.norm(r, p=2) for r in representations]
    return torch.var(torch.stack(norms))

def alignment_loss(representations, super_features):
    # Encourage similarity between each submodel's representation and the superlearner's
    loss = 0
    for r in representations:
        loss += 1 - torch.nn.functional.cosine_similarity(r, super_features, dim=-1).mean()

    return loss / len(representations)
def total_loss(y_hat, y_true, representations, feature_map, λ=0.1, μ=0.1):
    all_features = representations + [feature_map]
    l_cls = torch.nn.functional.cross_entropy(y_hat, y_true)
    l_balance = balance_loss(all_features)
    l_align = alignment_loss(representations, feature_map)
    return l_cls + λ * l_balance + μ * l_align
