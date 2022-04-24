from typing import Sequence, Tuple

import torch
from bunch import Bunch
from torch import Tensor


def transform_image(
    pixel_values: Sequence[str],
    input_dim: Tuple[int, ...],
) -> Tensor:
    normalized_pixel_values = torch.tensor([float(p) / 255.0 for p in pixel_values])
    if len(input_dim) > 1:
        input_dim_in_hwc = (input_dim[1], input_dim[2], input_dim[0])
        image_in_hwc = normalized_pixel_values.view(input_dim_in_hwc)
        image_in_chw = image_in_hwc.permute(2, 0, 1)
        image = image_in_chw
    else:
        image = normalized_pixel_values

    assert (image >= 0).all()
    assert (image <= 1).all()
    return image


def transform_and_bound(
    pixel_values: Sequence[str],
    config: Bunch,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Tensor, Tensor, Tensor]:
    image = transform_image(pixel_values, config.input_dim)
    input_lb = (image - config.eps).clamp(min=0)
    input_ub = (image + config.eps).clamp(max=1)
    try:
        means = torch.tensor(config.normalization_means).unsqueeze(1).unsqueeze(2)
        stds = torch.tensor(config.normalization_stds).unsqueeze(1).unsqueeze(2)
        image = normalize(image, means, stds)
        input_lb = normalize(input_lb, means, stds)
        input_ub = normalize(input_ub, means, stds)
    except AttributeError:
        pass  # no normalization needed
    return image.unsqueeze(0).to(device), input_lb.to(device), input_ub.to(device)


def normalize(image: Tensor, means: Tensor, stds: Tensor) -> Tensor:
    return (image - means) / stds
