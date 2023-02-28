from typing import Dict, List, Tuple

import dill  # type: ignore[import]
import torch
from torch import Tensor


class BilinearInterpol:
    def __init__(
        self,
        inner: Dict[Tuple[float, float], float],
        outer: Dict[Tuple[float, float], float],
        inner_range: float,
        outer_range: float,
        inner_res: float,
        outer_res: float,
    ) -> None:
        self.inner = inner
        self.outer = outer
        self.inner_range = inner_range
        self.outer_range = outer_range
        self.inner_res = inner_res
        self.outer_res = outer_res

    def get_value(self, lb: Tensor, ub: Tensor) -> Tensor:
        if abs(lb) < self.inner_range and abs(ub) < self.inner_range:
            res = self.inner_res
            tb = self.inner
        elif abs(lb) < self.outer_range and abs(ub) < self.outer_range:
            res = self.outer_res
            tb = self.outer
        else:  # At this point it is numerically irrellevant what we return
            return (lb + ub) / 2

        lb_low = (float(lb) / res) // 1 * res
        ub_low = (float(ub) / res) // 1 * res
        lb_high = lb_low + res
        ub_high = ub_low + res

        if ub_low <= lb_high:  # In one resolution interval
            return (lb + ub) / 2
        else:
            ll = tb[(lb_low, ub_low)]
            lh = tb[(lb_low, ub_high)]
            hl = tb[(lb_high, ub_low)]
            hh = tb[(lb_high, ub_high)]

            # interpolate
            r1 = (lb_high - lb) / res * ll + (lb - lb_low) / res * hl
            r2 = (lb_high - lb) / res * lh + (lb - lb_low) / res * hh
            y = (ub_high - ub) / res * r1 + (ub - ub_low) / res * r2
            return y

    @classmethod
    def create_from_data(
        cls,
        inner_values: List[Tuple[float, float, float]],
        outer_values: List[Tuple[float, float, float]],
        inner_range: float,
        outer_range: float,
        inner_res: float,
        outer_res: float,
    ) -> "BilinearInterpol":
        inner_dict: Dict[Tuple[float, float], float] = {}
        outer_dict: Dict[Tuple[float, float], float] = {}
        for lb, ub, c in inner_values:
            inner_dict[(float(lb), float(ub))] = float(c)
        for val in torch.linspace(
            -inner_range, inner_range, int(2 * inner_range / inner_res) + 1
        ):
            fv = float(val)
            inner_dict[(fv, fv)] = fv
        for lb, ub, c in outer_values:
            outer_dict[(float(lb), float(ub))] = float(c)
        for val in torch.linspace(
            -outer_range, outer_range, int(2 * outer_range / outer_range) + 1
        ):
            fv = float(val)
            outer_dict[(fv, fv)] = fv
        return cls(
            inner_dict, outer_dict, inner_range, outer_range, inner_res, outer_res
        )

    @classmethod
    def load_from_path(cls, path: str) -> "BilinearInterpol":
        try:
            with open(path, "rb") as file:
                interpol: "BilinearInterpol" = dill.load(file)
                return interpol
        except BaseException as e:
            print(f"Encountered exception during load {str(e)}")
            raise RuntimeError()

    def store_to_path(self, path: str) -> None:
        try:
            with open(path, "wb") as file:
                dill.dump(self, file)
        except BaseException as e:
            print(f"Encountered exception during store {str(e)}")
