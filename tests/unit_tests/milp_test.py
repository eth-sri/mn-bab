import torch
from gurobipy import GRB  # type: ignore[import]

from tests.gurobi_util import create_milp_model
from tests.test_util import get_deep_poly_bounds, toy_net


class TestAgainstMILP:
    """
    We compare our bounds with the bounds obtained by solving the verification with a MILP solver. The MILP bounds are exact.
    """

    def test_deep_poly_toy_example(self) -> None:
        network = toy_net()[0]
        input_lb = torch.tensor([-1.0, -1.0])
        input_ub = torch.tensor([1.0, 1.0])
        network.set_layer_bounds_via_interval_propagation(input_lb, input_ub)

        model, var_list = create_milp_model(network, input_lb, input_ub)

        output_node_var = var_list[-1]
        obj = output_node_var

        model.setObjective(obj, GRB.MINIMIZE)
        model.optimize()
        output_lb = model.objVal
        assert output_lb == 1

        model.setObjective(obj, GRB.MAXIMIZE)
        model.optimize()
        output_ub = model.objVal
        assert output_ub == 3

    def test_deep_poly_sound_on_toy_net(self) -> None:
        network = toy_net()[0]

        input_lb = torch.tensor([-1.0, -1.0])
        input_ub = torch.tensor([1.0, 1.0])

        (
            output_lb_deep_poly,
            output_ub_deep_poly,
        ) = get_deep_poly_bounds(network, input_lb.unsqueeze(0), input_ub.unsqueeze(0))

        network.set_layer_bounds_via_interval_propagation(input_lb, input_ub)
        model, var_list = create_milp_model(network, input_lb, input_ub)

        output_node_var = var_list[-1]
        obj = output_node_var

        model.setObjective(obj, GRB.MINIMIZE)
        model.optimize()
        output_lb_milp = model.objVal

        model.setObjective(obj, GRB.MAXIMIZE)
        model.optimize()
        output_ub_milp = model.objVal

        assert output_lb_deep_poly <= output_lb_milp
        assert output_ub_deep_poly >= output_ub_milp
