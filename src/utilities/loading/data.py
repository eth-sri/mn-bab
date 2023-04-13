from typing import Sequence, Tuple

import torch
from bunch import Bunch  # type: ignore[import]
from torch import Tensor


def transform_image(
    pixel_values: Sequence[str],
    input_dim: Tuple[int, ...],
) -> Tensor:
    if len(pixel_values)==9409:
        normalized_pixel_values = torch.tensor([float(p) for p in pixel_values[0:-1]])
    else:
        normalized_pixel_values = torch.tensor([float(p) / 255.0 for p in pixel_values])
    if len(input_dim) >= 3:
        if len(pixel_values)==9409:
            input_dim_in_chw = (input_dim[0], input_dim[1], input_dim[2])
            image_in_chw = normalized_pixel_values.view(input_dim_in_chw)
            image_dim = input_dim_in_chw
        else:
            input_dim_in_hwc = (input_dim[1], input_dim[2], input_dim[0])
            image_in_hwc = normalized_pixel_values.view(input_dim_in_hwc)
            image_in_chw = image_in_hwc.permute(2, 0, 1)
            image_dim = input_dim
        image = image_in_chw
    elif len(input_dim) > 1:
        image_dim = (int(torch.prod(torch.tensor(input_dim))),)
        image = normalized_pixel_values.view(input_dim)
    else:
        image = normalized_pixel_values
        image_dim = input_dim

    assert (image >= 0).all()
    assert (image <= 1).all()
    return image, image_dim


def transform_and_bound(
    pixel_values: Sequence[str],
    config: Bunch,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Tensor, Tensor, Tensor]:
    image, image_dim = transform_image(pixel_values, config.input_dim)
    input_lb = (image - config.eps).clamp(min=0)
    input_ub = (image + config.eps).clamp(max=1)
    try:
        means = torch.tensor(config.normalization_means)
        stds = torch.tensor(config.normalization_stds)
        if len(image_dim) == 3:
            means = means.unsqueeze(1).unsqueeze(2)
            stds = stds.unsqueeze(1).unsqueeze(2)
        else:
            assert len(means) == 1
            assert len(stds) == 1
        image = normalize(image, means, stds)
        input_lb = normalize(input_lb, means, stds)
        input_ub = normalize(input_ub, means, stds)
    except AttributeError:
        pass  # no normalization needed
    image = image.view(image_dim)
    input_lb = input_lb.view(image_dim)
    input_ub = input_ub.view(image_dim)
    return (
        image.unsqueeze(0).to(device),
        input_lb.unsqueeze(0).to(device),
        input_ub.unsqueeze(0).to(device),
    )


def normalize(image: Tensor, means: Tensor, stds: Tensor) -> Tensor:
    return (image - means) / stds
