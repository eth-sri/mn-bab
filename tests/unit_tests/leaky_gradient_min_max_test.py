import torch

from src.utilities.leaky_gradient_maximum_function import LeakyGradientMaximumFunction
from src.utilities.leaky_gradient_minimum_function import LeakyGradientMinimumFunction

leaky_gradient_minimum = LeakyGradientMinimumFunction.apply
leaky_gradient_maximum = LeakyGradientMaximumFunction.apply


class TestLeakyGradientMinMaxFunctions:
    def test_min_forward(self) -> None:
        smaller = torch.tensor(2.0)
        larger = torch.tensor(3.0)

        assert torch.minimum(smaller, larger) == leaky_gradient_minimum(smaller, larger)

    def test_min_backward(self) -> None:
        smaller = torch.tensor(2.0, requires_grad=True)
        larger = torch.tensor(3.0, requires_grad=True)
        min = leaky_gradient_minimum(smaller, larger)

        min.backward()
        assert smaller.grad == 1
        assert larger.grad == 1

    def test_max_forward(self) -> None:
        smaller = torch.tensor(2.0)
        larger = torch.tensor(3.0)
        assert torch.maximum(smaller, larger) == leaky_gradient_maximum(smaller, larger)

    def test_max_backward(self) -> None:
        smaller = torch.tensor(2.0, requires_grad=True)
        larger = torch.tensor(3.0, requires_grad=True)
        max = leaky_gradient_maximum(smaller, larger)

        max.backward()
        assert smaller.grad == 1
        assert larger.grad == 1
