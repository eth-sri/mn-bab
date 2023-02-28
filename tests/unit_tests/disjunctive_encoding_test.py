import torch

from src.mn_bab_verifier import MNBaBVerifier
from src.state.subproblem_state import SubproblemState
from src.utilities.argument_parsing import get_config_from_json
from src.utilities.config import make_config

# from src.utilities.config import AbstractDomain
from src.utilities.output_property_form import OutputPropertyForm
from src.verification_instance import get_asnet, get_io_constraints_from_spec


class TestDisjunctiveEncoding:
    """
    We test with our Disjunctive ReLU encoding implementation
    """

    def test_disjunctive_encoding_generation(self, n: int = 1) -> None:
        # Reads from the vnn spec for robot
        cfg_path = "configs/vnncomp22/vnn22_reach_robot.json"
        spec_path = (
            "vnn-comp-2022-sup/benchmarks/reach_prob_density/vnnlib/robot_0.vnnlib"
        )
        net_path = "vnn-comp-2022-sup/benchmarks/reach_prob_density/onnx/robot.onnx"

        bunch_config = get_config_from_json(cfg_path)
        config = make_config(**bunch_config)

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        config.use_gpu = torch.cuda.is_available()

        (inputs, input_constraints, target_constr,) = get_io_constraints_from_spec(
            spec_path=spec_path,
            config=config,
            device=device,
        )

        net, as_network = get_asnet(net_path, config, device)
        # Get verifier
        verifier = MNBaBVerifier(as_network, device, config.verifier)

        input_point = inputs[0]
        (input_lb, input_ub) = input_constraints[0]
        gt_constraint = target_constr[0]

        # Orig. constraint
        # y_1 > 0.0649 || y_1 < -0.0649 || y_2 > 0.0649 || y_2 < -0.0649 || y_0 < 0.155
        # We add the following constraints
        # y_0 > -50 || y_2 > -0.0649 (found by dp bounds)
        new_clause = [(0, -1, -50), (2, -1, 0.0649)]
        out_constr = [*gt_constraint, new_clause]

        # No batch dimension
        input_lb = input_lb.unsqueeze(0)
        input_ub = input_ub.unsqueeze(0)
        input_point = input_point.unsqueeze(0)

        out_prop_form = OutputPropertyForm.create_from_properties(
            out_constr,
            disjunction_adapter=None,
            use_disj_adapter=True,
            n_class=as_network.output_dim[-1],
            device=input_lb.device,
            dtype=input_lb.dtype,
        )

        assert len(out_prop_form.properties_to_verify) == 2
        assert (
            out_prop_form.property_matrix == torch.eye(2, device=out_prop_form.device)
        ).all()
        assert (out_prop_form.property_matrix == out_prop_form.combination_matrix).all()
        assert out_prop_form.disjunction_adapter is not None
        assert out_prop_form.disjunction_adapter[0].weight.shape == torch.Size([7, 5])
        assert out_prop_form.disjunction_adapter[0].bias.shape == torch.Size([7])
        assert out_prop_form.disjunction_adapter[2].weight.shape == torch.Size([2, 7])
        assert out_prop_form.disjunction_adapter[2].bias.shape == torch.Size([2])

        verifier.append_out_adapter(
            out_prop_form.disjunction_adapter,
            device=out_prop_form.device,
            dtype=out_prop_form.dtype,
        )

        (
            dp_out_prop_form,
            bounds,
            verified,
            falsified,
            ub_inputs,
        ) = verifier._verify_output_form_with_deep_poly(
            input_lb,
            input_ub,
            out_prop_form,
            compute_sensitivity=False,
            subproblem_state=SubproblemState.create_default(
                split_state=None,
                optimize_prima=False,
                batch_size=1,
                device=input_lb.device,
                use_params=False,
            ),
            ibp_pass=True,
        )

        assert len(dp_out_prop_form.properties_to_verify) == 1


if __name__ == "__main__":
    T = TestDisjunctiveEncoding()
    T.test_disjunctive_encoding_generation()
