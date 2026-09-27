import math

import torch
import torch.nn.functional as F


def mrn_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    components: int,
    eps: float = 1e-8,
    normalize: bool = True,
) -> torch.Tensor:
    x, y = torch.broadcast_tensors(x, y)
    total_dim = x.shape[-1]
    if total_dim % components != 0:
        raise ValueError(
            f"Embedding dim {total_dim} must be divisible by components {components}."
        )

    component_dim = total_dim // components
    x_split = x.reshape(*x.shape[:-1], components, component_dim)
    y_split = y.reshape(*y.shape[:-1], components, component_dim)
    diff = x_split - y_split

    asymmetric_dim = component_dim // 2
    if asymmetric_dim > 0:
        max_component = F.relu(diff[..., :asymmetric_dim].amax(dim=-1))
    else:
        max_component = torch.zeros_like(diff[..., 0])

    l2_part = diff[..., asymmetric_dim:]
    if l2_part.shape[-1] > 0:
        l2_component = torch.linalg.norm(l2_part + eps, dim=-1)
    else:
        l2_component = torch.zeros_like(max_component)

    distance = (max_component + l2_component).mean(dim=-1)
    if normalize:
        distance = distance / math.sqrt(float(total_dim))
    return distance


def alignment_score(
    x: torch.Tensor,
    y: torch.Tensor,
    components: int,
) -> torch.Tensor:
    return torch.exp(-mrn_distance(x, y, components=components))
