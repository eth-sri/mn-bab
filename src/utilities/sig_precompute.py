from typing import Callable, List, Tuple, Union

import torch
from torch import Tensor

from src.abstract_layers.abstract_sig_base import SigBase
from src.abstract_layers.abstract_sigmoid import d_sig, sig
from src.abstract_layers.abstract_tanh import d_tanh, tanh
from src.utilities.bilinear_interpolator import BilinearInterpol


def approx_diff_integral(
    l1: Tensor,
    b1: Tensor,
    f2: Callable[[Tensor], Tensor],
    lb: float,
    ub: float,
    steps: int = 1000,
) -> Tensor:
    x = torch.linspace(lb, ub, steps)
    y = l1 * x + b1
    y_2 = f2(x)
    return torch.abs(y - y_2).sum() * (ub - lb) / steps


# Top be set in main
act: Callable[[Tensor], Tensor] = sig
d_act: Callable[[Tensor], Tensor] = d_sig


def get_best_split_for_bounds(
    lb: float, ub: float, tangent_points: Tensor, steps: int = 200
) -> float:
    poss_c = torch.linspace(lb, ub, steps)
    lbs = lb * torch.ones_like(poss_c)
    ubs = ub * torch.ones_like(poss_c)

    # Get bounds
    (
        lb_slope_l,
        ub_slope_l,
        lb_intercept_l,
        ub_intercept_l,
    ) = SigBase._get_approximation_slopes_and_intercepts_for_act(
        bounds=(lbs, poss_c),
        tangent_points=tangent_points,
        step_size=0.01,
        max_x=500,
        act=act,
        d_act=d_act,
    )
    (
        lb_slope_u,
        ub_slope_u,
        lb_intercept_u,
        ub_intercept_u,
    ) = SigBase._get_approximation_slopes_and_intercepts_for_act(
        bounds=(poss_c, ubs),
        tangent_points=tangent_points,
        step_size=0.01,
        max_x=500,
        act=act,
        d_act=d_act,
    )

    best_c = -1
    best_v: Union[float, Tensor] = torch.inf

    for i, c in enumerate(poss_c):
        lower_cont_l = approx_diff_integral(
            lb_slope_l[i], lb_intercept_l[i], act, lb, c
        )
        lower_cont_u = approx_diff_integral(
            ub_slope_l[i], ub_intercept_l[i], act, lb, c
        )
        upper_cont_l = approx_diff_integral(
            lb_slope_u[i], lb_intercept_u[i], act, c, ub
        )
        upper_cont_u = approx_diff_integral(
            ub_slope_u[i], ub_intercept_u[i], act, c, ub
        )
        diff = lower_cont_l + lower_cont_u + upper_cont_l + upper_cont_u
        # print(f"{i}: c- {c} diff - {diff}")
        if diff < best_v:
            best_c = c
            best_v = diff

    return best_c


def get_training_data(
    lb: float, ub: float, steps: int, tangent_points: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    bound_l: List[float] = []
    bound_u: List[float] = []
    centers: List[float] = []
    for cub in torch.linspace(lb, ub, steps):
        for clb in torch.linspace(lb, ub, steps):
            if clb >= cub:
                continue
            c = get_best_split_for_bounds(clb, cub, tangent_points, steps=500)
            bound_l.append(clb)
            bound_u.append(cub)
            centers.append(c)
            print(f"LB {clb} UB {cub} Center {c}")
    return torch.Tensor(bound_l), torch.Tensor(bound_u), torch.Tensor(centers)


if __name__ == "__main__":

    generate_data = False
    generate_interpol = True
    eval_interpol = False
    use_sig = True
    # Set activation and derivative
    if use_sig:
        act = sig
        d_act = d_sig
    else:
        act = tanh
        d_act = d_tanh
    name = str(act).split(" ")[1]

    (
        intersection_points,
        tangent_points,
        step_size,
        max_x,
    ) = SigBase._compute_bound_to_tangent_point(act, d_act)
    if generate_data:
        bound_l, bound_u, centers = get_training_data(-495, 495, 199, tangent_points)
        torch.save(bound_l, f"./data/{name}_bound_l_495_5.pt")
        torch.save(bound_u, f"./data/{name}_bound_u_495_5.pt")
        torch.save(centers, f"./data/{name}_centers_495_5.pt")
        bound_l, bound_u, centers = get_training_data(-15, 15, 61, tangent_points)
        torch.save(bound_l, f"./data/{name}_bound_l_15.pt")
        torch.save(bound_u, f"./data/{name}_bound_u_15.pt")
        torch.save(centers, f"./data/{name}_centers_15.pt")
    if generate_interpol:
        store_at_path = f"./data/{name}_bil_interpol.pkl"
        inner_l = torch.load(f"./data/{name}_bound_l_15.pt")
        inner_u = torch.load(f"./data/{name}_bound_u_15.pt")
        inner_c = torch.load(f"./data/{name}_centers_15.pt")
        inner = [(inner_l[i], inner_u[i], inner_c[i]) for i in range(len(inner_l))]
        outer_l = torch.load(f"./data/{name}_bound_l_495_5.pt")
        outer_u = torch.load(f"./data/{name}_bound_u_495_5.pt")
        outer_c = torch.load(f"./data/{name}_centers_495_5.pt")
        outer = [(outer_l[i], outer_u[i], outer_c[i]) for i in range(len(outer_l))]

        interpol = BilinearInterpol.create_from_data(inner, outer, 15, 495, 0.5, 5)
        interpol.store_to_path(store_at_path)
    if eval_interpol:
        load_from_path = f"./data/{name}_bil_interpol.pkl"
        interpol = BilinearInterpol.load_from_path(load_from_path)
        # plot_points_3d(bound_l, bound_u, centers)
        # plot_points_fix_ub(-200, 15)
        # diffs = []
        # rel_diffs = []
        # for ub in torch.linspace(-14, 14, 200):
        #     for lb in torch.linspace(-14, 14, 200):
        #         if lb >= ub:
        #             continue
        #         start_time = time.time()
        #         opt_c = get_best_split_for_bounds(lb, ub, tangent_points, steps=100)
        #         total_time = time.time() - start_time
        #         start_time = time.time()
        #         reg_c = interpol.get_value(lb, ub)
        #         total_time2 = time.time() - start_time
        #         diffs.append(abs(opt_c - reg_c))
        #         if abs(opt_c - reg_c) > 100:
        #             reg_c = interpol.get_value(lb, ub)
        #         rel_diffs.append(abs(opt_c - reg_c) / (ub - lb))
        #         print(
        #             f"Opt: {opt_c} Diff: {abs(opt_c-reg_c)} Time: {total_time} Reg: {reg_c} Time: {total_time2}"
        #         )
        # print("============")
        # diffs = torch.Tensor(diffs)
        # rel_diffs = torch.Tensor(rel_diffs)
        # print(f"Avg: {diffs.sum()/diffs.numel()} Max: {torch.max(diffs)}")
        # print(
        #     f"Rel. Avg: {rel_diffs.sum()/rel_diffs.numel()} Max: {torch.max(rel_diffs)}"
        # )


# def plot_approx(l1_l: float, b1_l: float, l1_u: float, b1_u: float, l2_l: float, b2_l:float, l2_u: float, b2_u:float, f2: Callable[[Tensor], Tensor], lb: float, c: float, ub: float, steps: int = 1000):
#     x1 = torch.linspace(lb, c, steps)
#     x2 = torch.linspace(c, ub, steps)
#     y1_l = l1_l*x1+b1_l
#     y1_u = l1_u*x1+b1_u
#     y2_l = l2_l*x2+b2_l
#     y2_u = l2_u*x2+b2_u
#     y1 = f2(x1)
#     y2 = f2(x2)

#     plt.plot(x1, y1_l, label = "Linear Lower")
#     plt.plot(x1, y1_u, label = "Linear Upper")
#     plt.plot(x1, y1, label = "Sigmoid")
#     plt.plot(x2, y2_l, label = "Linear Lower")
#     plt.plot(x2, y2_u, label = "Linear Upper")
#     plt.plot(x2, y2, label = "Sigmoid")
#     plt.show()

# def plot_points_3d(x1: Tensor, x2: Tensor, y: Tensor):
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     ax.scatter(x1,x2,y)
#     plt.show()

# def plot_points_fix_ub(lb:float, ub: float):

#     bound_l_select = []
#     center_select = []
#     for lb in torch.linspace(lb, ub, 1000):
#         opt_c = get_best_split_for_bounds(lb, ub, tangent_points, steps=500)
#         bound_l_select.append(lb)
#         center_select.append(opt_c)
#     bound_l_select = torch.Tensor(bound_l_select)
#     center_select = torch.Tensor(center_select)
#     plt.plot(bound_l_select,center_select)
#     plt.title(f"Upper_bound = {ub}")
#     plt.show()
