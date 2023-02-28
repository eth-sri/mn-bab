from __future__ import annotations

from abc import ABC
from copy import copy
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bunch import Bunch  # type: ignore[import]

from src.state.tags import LayerTag


def _check_config(config: Any, bunch: Bunch, what: str) -> None:
    for k in bunch.keys():
        if k not in config.__dict__:
            raise ValueError("'{}' is not a valid {}".format(k, what))


# TODO: is all the boilerplate really necessary? (is there a way to get all declared field names before initialization?)


class LoggerConfig:
    # To view experiments live in comet, add the api key, workspace, and project name
    comet_options: Dict[str, Any]

    def __init__(self, config: Bunch):
        self.comet_options = {
            "api_key": config.comet_api_key,
            "project_name": config.comet_project_name,
            "workspace": config.comet_workspace,
            "disabled": not config.use_online_logging,
        }


class NetworkConfig:
    path: str
    n_layers: Optional[int]  # Number of layers of a fully connected network.
    n_neurons_per_layer: Optional[
        int
    ]  # Number of neurons per hidden layer in fully connected network. Used to construct network when only weights are provided.

    def __init__(self, config: Bunch):
        self.path = config.get("network_path", "")
        try:
            self.n_layers = config.n_layers
            self.n_neurons_per_layer = config.n_neurons_per_layer
        except AttributeError:
            self.n_layers = None
            self.n_neurons_per_layer = None

    def load_params(self) -> Dict[str, Any]:  # TODO: use TypedDict?
        return self.__dict__


class MNBabParameterBaseConfig(ABC):
    optimize: bool
    lr: float
    opt_iterations: int


class MNBabAlphaConfig(MNBabParameterBaseConfig):
    def __init__(self, config: Bunch):
        self.optimize = config.get(
            "optimize_alpha", False
        )  # Whether to optimize relaxation slopes
        self.lr = config.get("alpha_lr", 0.1)  # LR used to optimize relaxation slopes
        self.opt_iterations = config.get(
            "alpha_opt_iterations", 20
        )  # Nr. of iterations of pure slope optimization only used on root subproblem or when prima optimization is turned off
        self.bab_opt_iterations = config.get(
            "bab_alpha_opt_iterations", self.opt_iterations
        )  # Nr. of iterations of pure slope optimization in BaB


class MNBabBetaConfig:
    lr: float

    def __init__(self, config: Bunch):
        self.lr = config.get(
            "beta_lr", 0.05
        )  # LR used to optimize lagrange parameters for bound enforcement


class PrimaGroupRandomness(Enum):
    none = "none"  # Use sparse heuristic
    only = "only"  # Use purely random groups sampled uniformly from K-Tuples
    augment = (
        "augment"  # Augment the sparse heuristic with "off-diagonal" random groups
    )


class PrimaHyperparameters:
    sparse_n: int  # Partition size for initial split of neurons for multi neuron constraint computation
    K: int  # Neuron group size for MNC computation. Set to 3 or 4
    s: int  # Maximum permitted overlap between neuron groups for MNC computation. Set to K-1 to use all groups (not recommended).
    num_proc_to_compute_constraints: int  # Number of parallel processes to use to compute MNC constraints
    max_number_of_parallel_input_constraint_queries: int  # Maximum number of octahedral constraints evaluated at once. Only limited by GPU memory.
    max_unstable_nodes_considered_per_layer: int  # Only consider this many neurons for MNC computation per layer, sorted by largest area in 2d input-output space
    min_relu_transformer_area_to_be_considered: float  # Only consider neurons for MNC computation per layer with at least this area in 2d input-output space (no MNC for almost stable neurons)
    fraction_of_constraints_to_keep: float  # Use box approximation to compute the worst possible constraint violation. Only keep this portion of constraints sorted by worst maximum violation
    random_prima_groups: PrimaGroupRandomness  # "none" to only use prima groups produced by our sparsity heuristic, "only" to only use uniformly at random sampled k-tuples and "augment" to add "off-diagonal" groupings.
    prima_sparsity_factor: float  # Determines total number of prima constraints as a factor of the number obtained via our heuristic. < 1 is always respected, >1 only in combination with "only" or "augment" random groups

    def __init__(self, bunch: Bunch):
        self.sparse_n = bunch.get("sparse_n", 50)
        self.K = bunch.get("K", 3)
        self.s = bunch.get("s", 1)
        self.num_proc_to_compute_constraints = bunch.get(
            "num_proc_to_compute_constraints", 2
        )
        self.max_number_of_parallel_input_constraint_queries = bunch.get(
            "max_number_of_parallel_input_constraint_queries", 10000
        )
        self.max_unstable_nodes_considered_per_layer = bunch.get(
            "max_unstable_nodes_considered_per_layer", 1000
        )
        self.min_relu_transformer_area_to_be_considered = bunch.get(
            "min_relu_transformer_area_to_be_considered", 0.05
        )
        # TODO: How is this handled for sigmoidal layers?
        self.fraction_of_constraints_to_keep = bunch.get(
            "fraction_of_constraints_to_keep", 1.0
        )
        self.random_prima_groups = PrimaGroupRandomness(
            bunch.get("random_prima_groups", "none")
        )
        self.prima_sparsity_factor = bunch.get("prima_sparsity_factor", 1.0)


def make_prima_hyperparameters(
    **hyperparameters: Dict[str, Any]
) -> PrimaHyperparameters:
    bunch = Bunch(**hyperparameters)
    config = PrimaHyperparameters(bunch)
    _check_config(config, bunch, "prima hyperparameter")
    return config


class MNBabPrimaConfig(MNBabParameterBaseConfig):
    hyperparameters: PrimaHyperparameters

    def __init__(self, config: Bunch):
        self.optimize = config.get(
            "optimize_prima", False
        )  # Whether to use multi neuron constraints
        self.lr = config.get(
            "prima_lr", 0.01
        )  # LR to optimize the lagrange parameters associated with MNCs
        self.opt_iterations = config.get(
            "prima_opt_iterations", 20
        )  # Number of iterations for prima constraint optimization. Use for the whole BaB process
        self.hyperparameters = PrimaHyperparameters(
            config.get("prima_hyperparameters", Bunch())
        )
        self.bab_opt_iterations = config.get(
            "bab_prima_opt_iterations", self.opt_iterations
        )  # Nr. of iterations of pure slope optimization in BaB


class LearningRateConfig:
    peak_scaling_factor: float  # LR is scaled up to this in a 1cycle schedule
    final_div_factor: float  # LR is decreased to this in a 1cycle schedule

    def __init__(self, config: Bunch):
        self.peak_scaling_factor = config.get("peak_lr_scaling_factor", 2.0)
        self.final_div_factor = config.get("final_lr_div_factor", 1e1)


class LayerType(Enum):
    fully_connected = "fully_connected"
    conv2d = "conv2d"


class ParameterSharing(Enum):
    same_layer = "same_layer"  # share parameters across all queries originating in the same layer (default behavior)
    none = "none"  # use individual parameters for each component of the bound
    in_channel = "in_channel"  # share parameters for neurons in the same channel (makes sense for conv2d layers)


class ParameterSharingConfig:
    entries: Sequence[Tuple[LayerType, ParameterSharing]]

    @property
    def reduce_parameter_sharing(self) -> bool:
        return any(
            parameter_sharing != ParameterSharing.same_layer
            for (layer_type, parameter_sharing) in self.entries
        )

    def __init__(self, config: Bunch):
        self.entries = [(LayerType(k), ParameterSharing(v)) for k, v in config.items()]


class LayerFilter:
    code: str

    def __init__(self, code: str):
        self.code = code

    def filter_layer_ids(self, layer_ids: Sequence[LayerTag]) -> Sequence[LayerTag]:
        return eval(self.code)


class IntermediateBoundOptimizationConfig:
    prima_lr: float
    alpha_lr: float
    lr_config: LearningRateConfig  # The LearningRate Config
    num_layers_to_optimize: int  # Number of layers to optimize (from the front)
    optimization_iterations: int  # Number of optimization iterations
    optimize_prior_bounds: bool  # Whether to also parameterize the lower and upper bounds of prior activation nodes (instead of only the slopes)
    indiv_optim: bool  # Whether to optimize each neuron individually
    adapt_optim: bool  # Whether to optimize an adaptive loss

    def __init__(self, config: Bunch):
        self.prima_lr = config.get("prima_lr", 0.01)
        self.alpha_lr = config.get("alpha_lr", 0.1)
        self.lr_config = LearningRateConfig(config.get("lr_config", Bunch()))
        self.num_layers_to_optimize = config.get("num_layers_to_optimize", float("inf"))
        self.optimization_iterations = config.get("optimization_iterations", 30)
        self.optimize_prior_bounds = config.get("optimize_prior_bounds", False)
        self.indiv_optim = config.get("indiv_optim", False)
        self.adapt_optim = config.get("adapt_optim", False)
        assert not (
            self.indiv_optim and self.adapt_optim
        ), "Invalid Intermediate Bound Config"


class ReLUAlphaInitMethod(Enum):
    minimum_area = "minimum_area"  # use DeepPoly heuristic (relaxation slope initialized to either 0 or 1)
    one_half = "one_half"  # always initialize relaxation slopes to 0.5


class IntermediateBoundsMethod(Enum):
    none = 0
    dp = 1
    alpha = 2
    prima = 3

    def __lt__(self, other: IntermediateBoundsMethod) -> bool:
        if self.__class__ is other.__class__:
            return self.value < other.value
        else:
            raise NotImplementedError

    def __le__(self, other: IntermediateBoundsMethod) -> bool:
        if self.__class__ is other.__class__:
            return self.value <= other.value
        else:
            raise NotImplementedError


class BacksubstitutionConfig:
    use_dependence_sets: bool  # Instead of computing the sensitivity wrt the whole preceeding layer, only compute the receptive field. Only applied to conv layers and precludes use of PRIMA. Significant memory savings.
    use_early_termination: bool  # Terminate backpropagation early if all querries are "stable", i.e. property can be shown. Only use for ReLU networks.
    intermediate_bounds_method: IntermediateBoundsMethod  # (Most precise) method to use to compute intermediate bounds
    box_pass: bool  # Whether to do a box propagation before DP passes to determine stable neurons
    domain_pass: AbstractDomain  # Use a fwd pass with the abstract domain specified here to set bounds prior to bounding the root subproblem
    relu_alpha_init_method: ReLUAlphaInitMethod  # How to initialize alpha parameters (relaxation slope) for the ReLU activation function.
    max_num_queries: Optional[int]  # Maximum number of queries used at the same time
    intermediate_bound_optimization_config: IntermediateBoundOptimizationConfig  # Configures the intermediate bound optimization during the backsubstitution

    parameter_sharing_config: Optional[
        ParameterSharingConfig
    ]  # Configures how parameters are shared among intermediate bound queries
    layer_ids_for_which_to_reduce_parameter_sharing: Sequence[LayerTag]

    prima_hyperparameters: Optional[PrimaHyperparameters]
    layer_ids_for_which_to_compute_prima_constraints: Sequence[LayerTag]

    @property
    def reduce_parameter_sharing(self) -> bool:
        return (
            self.parameter_sharing_config is not None
            and self.parameter_sharing_config.reduce_parameter_sharing
        )

    @property
    def optimize_prima(self) -> bool:
        return self.prima_hyperparameters is not None

    def _assert_compatible(self) -> None:
        assert not (
            self.use_dependence_sets and self.prima_hyperparameters is not None
        ), "dependence sets and prima constraints are incompatible because prima constraints introduce additional dependencies"

    def __init__(self, config: Bunch):
        self.use_dependence_sets = config.get("use_dependence_sets", False)
        self.use_early_termination = config.get("use_early_termination", False)
        self.box_pass = config.get("box_pass", False)
        self.domain_pass = AbstractDomain(config.get("domain_pass", "none"))
        self.intermediate_bounds_method = IntermediateBoundsMethod[
            config.get("intermediate_bounds_method", "prima")
        ]
        self.relu_alpha_init_method = ReLUAlphaInitMethod(
            config.get("relu_alpha_init_method", "minimum_area")
        )
        self.max_num_queries = config.get("max_num_queries", None)
        self.intermediate_bound_optimization_config = (
            IntermediateBoundOptimizationConfig(
                config.get("intermediate_bound_optimization_config", Bunch())
            )
        )

        self.parameter_sharing_config = None  # no reduced parameter sharing by default
        self.layer_ids_for_which_to_reduce_parameter_sharing = (
            []
        )  # no reduced parameter sharing by default

        self.prima_hyperparameters = None  # no prima by default
        self.layer_ids_for_which_to_compute_prima_constraints = (
            []
        )  # no prima by default
        self._assert_compatible()

    def where(  # TODO: implement this automatically somehow
        self,
        use_dependence_sets: Optional[bool] = None,
        use_early_termination: Optional[bool] = None,
        intermediate_bounds_method: Optional[IntermediateBoundsMethod] = None,
        relu_alpha_init_method: Optional[ReLUAlphaInitMethod] = None,
        max_num_queries: Optional[int] = -1,
        intermediate_bound_optimization_config: Optional[
            IntermediateBoundOptimizationConfig
        ] = None,
        # TODO: ParameterSharingConfig?
        layer_ids_for_which_to_reduce_parameter_sharing: Optional[
            Sequence[LayerTag]
        ] = None,
        # TODO: prima_hyperparameters?
        layer_ids_for_which_to_compute_prima_constraints: Optional[
            Sequence[LayerTag]
        ] = None,
        prima_hyperparameters: Optional[PrimaHyperparameters] = None,
    ) -> BacksubstitutionConfig:
        # TODO: manually assigning fields does not really scale
        result = BacksubstitutionConfig(Bunch())
        result.use_dependence_sets = (
            use_dependence_sets
            if use_dependence_sets is not None
            else self.use_dependence_sets
        )
        result.use_early_termination = (
            use_early_termination
            if use_early_termination is not None
            else self.use_early_termination
        )
        result.intermediate_bounds_method = (
            intermediate_bounds_method
            if intermediate_bounds_method is not None
            else self.intermediate_bounds_method
        )
        result.relu_alpha_init_method = (
            relu_alpha_init_method
            if relu_alpha_init_method is not None
            else self.relu_alpha_init_method
        )
        result.max_num_queries = (
            max_num_queries
            if max_num_queries != -1  # As None is a valid value
            else self.max_num_queries
        )
        result.intermediate_bound_optimization_config = (
            intermediate_bound_optimization_config
            if intermediate_bound_optimization_config is not None
            else self.intermediate_bound_optimization_config
        )

        result.parameter_sharing_config = self.parameter_sharing_config
        result.layer_ids_for_which_to_reduce_parameter_sharing = (
            layer_ids_for_which_to_reduce_parameter_sharing
            if layer_ids_for_which_to_reduce_parameter_sharing is not None
            else self.layer_ids_for_which_to_reduce_parameter_sharing
        )

        result.prima_hyperparameters = (
            prima_hyperparameters
            if prima_hyperparameters is not None
            else self.prima_hyperparameters
        )
        result.layer_ids_for_which_to_compute_prima_constraints = (
            layer_ids_for_which_to_compute_prima_constraints
            if layer_ids_for_which_to_compute_prima_constraints is not None
            else self.layer_ids_for_which_to_compute_prima_constraints
        )
        # result._assert_compatible()
        return result

    def with_parameter_sharing(
        self,
        parameter_sharing_config: ParameterSharingConfig,
        layer_ids_for_which_to_reduce_parameter_sharing: Sequence[LayerTag],
    ) -> BacksubstitutionConfig:
        result = copy(self)
        result.parameter_sharing_config = parameter_sharing_config
        result.layer_ids_for_which_to_reduce_parameter_sharing = (
            layer_ids_for_which_to_reduce_parameter_sharing
        )
        result._assert_compatible()
        return result

    def with_prima(
        self,
        prima_hyperparameters: PrimaHyperparameters,
        layer_ids_for_which_to_compute_prima_constraints: Sequence[LayerTag],
    ) -> BacksubstitutionConfig:
        # TODO: manually assigning fields does not really scale
        result = copy(self)
        result.prima_hyperparameters = prima_hyperparameters
        result.layer_ids_for_which_to_compute_prima_constraints = (
            layer_ids_for_which_to_compute_prima_constraints
        )
        # if self.use_dependence_sets:
        #     result.prima_hyperparameters = None
        # result._assert_compatible()
        return result

    def without_prima(
        self,
    ) -> BacksubstitutionConfig:
        result = copy(self)
        result.prima_hyperparameters = None
        result.layer_ids_for_which_to_compute_prima_constraints = []
        return result


def make_backsubstitution_config(**config: Any) -> BacksubstitutionConfig:
    bunch = Bunch(**config)
    return BacksubstitutionConfig(
        bunch
    )  # TODO: check spelling of configuration options


class MNBabOptimizerConfig:
    alpha: MNBabAlphaConfig
    beta: MNBabBetaConfig
    prima: MNBabPrimaConfig
    lr: LearningRateConfig

    parameter_sharing_config: ParameterSharingConfig  # TODO: make this configurable per parameter type?
    parameter_sharing_layer_id_filter: Optional[LayerFilter]

    def __init__(self, config: Bunch):
        self.alpha = MNBabAlphaConfig(config)
        self.beta = MNBabBetaConfig(config)
        self.prima = MNBabPrimaConfig(config)

        assert not (
            self.prima.optimize and not self.alpha.optimize
        ), "If you optimize prima constraints, you also have to optimize alpha."

        self.lr = LearningRateConfig(config)

        self.parameter_sharing_config = ParameterSharingConfig(
            config.get("parameter_sharing", Bunch())
        )

        self.parameter_sharing_layer_id_filter = (
            (LayerFilter(config["parameter_sharing_layer_id_filter"]))
            if "parameter_sharing_layer_id_filter" in config
            else None
        )

    def max_lr(self) -> List[float]:
        scaling = self.lr.peak_scaling_factor
        return [
            scaling * self.alpha.lr,
            scaling * self.beta.lr,
            scaling * self.prima.lr,
        ]


def make_optimizer_config(**config: Any) -> MNBabOptimizerConfig:
    bunch = Bunch(**config)
    return MNBabOptimizerConfig(bunch)  # TODO: check spelling of configuration options


class BranchingMethod(Enum):
    babsr = "babsr"  # Use Branch and Bound Smart ReLU heuristic (https://arxiv.org/abs/1909.06588)
    active_constraint_score = (
        "active_constraint_score"  # Use a MNC based heuristic, only use with PRIMA
    )
    filtered_smart_branching = "filtered_smart_branching"  # Evaluate the top k choices as per branching heuristic and take the best one


class PropagationEffectMode(Enum):
    none = "none"  # Ignore the propagation effect
    bias = "bias"  # Consider the minimum propagation effect
    intermediate_concretization = (
        "intermediate_concretization"  # Consider the maximum propagation effect
    )


class ReduceOp(Enum):
    min = "min"
    max = "max"
    geo_mean = "geo_mean"


class BranchingConfig(ABC):
    method: BranchingMethod  # Which heuristic to use to chose the neuron to branch on
    use_cost_adjusted_scores: bool  # Correct the branching scores with the expected cost for the split

    def babsr(self) -> BaBsrBranchingConfig:
        raise ValueError(
            "branching method is '{}', not 'babsr'".format(self.method.value)
        )

    def active_constraint_score(self) -> ActiveConstraintScoreBranchingConfig:
        raise ValueError(
            "branching method is '{}', not 'active_constraint_score'".format(
                self.method.value
            )
        )

    def filtered_smart_branching(self) -> FilteredSmartBranchingConfig:
        raise ValueError(
            "branching method is '{}', not 'filtered_smart_branching'".format(
                self.method.value
            )
        )


class BaBsrBranchingConfig(BranchingConfig):
    # We generalize the idea behind BaBSR to actually reflect our method better
    use_prima_contributions: bool  # Consider the MNC contribution when estimating bound improvement
    use_optimized_slopes: bool  # Use the actual slopes when estimating bound improvement
    use_beta_contributions: bool  # Use the beta contribution when estimating the bound improvement
    propagation_effect_mode: PropagationEffectMode  # How to consider the propagation effect when estimating the bound improvement
    use_indirect_effect: bool  # Compute the indirect effect of a split vie the improvement of other bounds
    reduce_op: ReduceOp  # How to aggregate the score obtained for the positive and negative split into one score
    use_abs: bool  # Consider all estimated changes as improvements, perhaps a better measure of sensitivity

    def __init__(self, config: Bunch):
        self.method = BranchingMethod.babsr
        self.use_prima_contributions = config.get("use_prima_contributions", False)
        self.use_optimized_slopes = config.get("use_optimized_slopes", False)
        self.use_beta_contributions = config.get("use_beta_contributions", False)
        self.propagation_effect_mode = PropagationEffectMode(
            config.get("propagation_effect_mode", "bias")
        )
        self.use_indirect_effect = config.get("use_indirect_effect", False)
        self.reduce_op = ReduceOp(config.get("reduce_op", "min"))
        self.use_abs = config.get("use_abs", True)
        self.use_cost_adjusted_scores = config.get("use_cost_adjusted_scores", False)

    def babsr(self) -> BaBsrBranchingConfig:
        return self


class ActiveConstraintScoreBranchingConfig(BranchingConfig):
    def __init__(self, config: Bunch):
        self.method = BranchingMethod.active_constraint_score
        self.use_cost_adjusted_scores = config.get("use_cost_adjusted_scores", False)

    def active_constraint_score(self) -> ActiveConstraintScoreBranchingConfig:
        return self


class FilteredSmartBranchingConfig(BranchingConfig):
    n_candidates: int  # The top how many candidates to evaluate using a DeepPoly pass
    reduce_op: ReduceOp  # How to aggregate the score obtained for the positive and negative split into one score

    def __init__(self, config: Bunch):
        self.method = BranchingMethod.filtered_smart_branching
        self.n_candidates = config.n_candidates
        self.reduce_op = ReduceOp(config.get("reduce_op", "min"))  # TODO: default ok?
        self.use_cost_adjusted_scores = config.get("use_cost_adjusted_scores", False)

    def filtered_smart_branching(self) -> FilteredSmartBranchingConfig:
        return self


def make_branching_config(**config: Any) -> BranchingConfig:
    method = BranchingMethod(config.get("method", "babsr"))

    result: BranchingConfig
    if method == BranchingMethod.babsr:
        result = BaBsrBranchingConfig(Bunch(**config))
    elif method == BranchingMethod.active_constraint_score:
        result = ActiveConstraintScoreBranchingConfig(Bunch(**config))
    elif method == BranchingMethod.filtered_smart_branching:
        result = FilteredSmartBranchingConfig(Bunch(**config))
    else:
        raise RuntimeError("Branching method misspecified.")
    _check_config(result, config, "branching method configuration option")
    return result


class BranchAndBoundConfig:
    batch_sizes: Sequence[
        int
    ]  # How many subproblems to consider in a batch depending on the layer at which we split. The larger the faster, limited by memory
    branching_config: BranchingConfig
    recompute_intermediate_bounds_after_branching: bool  # Whether to recompute the intermediate bounds of neurons ocuring after the split layer. Makes branching significantly more expensive.
    run_BaB: bool  # Whether to run BaB

    def __init__(self, config: Bunch):
        self.batch_sizes = config.get("bab_batch_sizes", [1])
        self.branching_config = make_branching_config(
            **config.get("branching", Bunch())
        )
        self.recompute_intermediate_bounds_after_branching = config.get(
            "recompute_intermediate_bounds_after_branching", False
        )
        self.run_BaB = config.get("run_BaB", True)


class MILPConfig:
    refine_via_milp: int  # Number of activation layers to refine via MILP
    refine_only_unstable: bool  # Whether to only refine bounds of unstable ReLUs
    pre_refine_via_ab_prima: bool
    timeout_refine_total: Optional[
        int
    ]  # Total amount of time we want to spend refining neurons, None is all time
    timeout_refine_neuron: Optional[
        int
    ]  # How much time we want to spend refining each neuron, None is all time
    solve_via_milp: bool

    def __init__(self, config: Bunch) -> None:
        milp_cfg = Bunch(config.get("milp", Bunch()))
        self.refine_via_milp = milp_cfg.get("refine_via_milp", 0)
        self.refine_only_unstable = milp_cfg.get("refine_only_unstable", True)
        self.pre_refine_via_ab_prima = milp_cfg.get("pre_refine_via_ab_prima", True)
        self.timeout_refine_total = milp_cfg.get("timeout_refine_total", None)
        self.timeout_refine_neuron = milp_cfg.get("timeout_refine_neuron", None)
        self.solve_via_milp = milp_cfg.get("solve_via_milp", False)


class OuterVerifierConfig:
    initial_dp: bool
    adversarial_attack: bool
    adversarial_attack_restarts: int
    refine_intermediate_bounds: bool
    refine_intermediate_bounds_prima: bool
    forward_dp_pass: bool
    milp_config: MILPConfig
    input_domain_splitting: bool
    simplify_onnx: bool
    use_disj_adapter: bool

    instance_pre_filter_batch_size: Optional[int]

    def __init__(self, config: Bunch):
        outer_config = Bunch(config.get("outer_verifier", Bunch()))
        self.initial_dp = outer_config.get("initial_dp", True)
        self.adversarial_attack = outer_config.get("adversarial_attack", True)
        self.adversarial_attack_restarts = outer_config.get(
            "adversarial_attack_restarts", 5
        )
        self.refine_intermediate_bounds = outer_config.get(
            "refine_intermediate_bounds", False
        )
        self.refine_intermediate_bounds_prima = outer_config.get(
            "refine_intermediate_bounds_prima", False
        )
        self.forward_dp_pass = outer_config.get("forward_dp_pass", False)
        self.milp_config = MILPConfig(outer_config)
        self.input_domain_splitting = outer_config.get("input_domain_splitting", False)
        self.simplify_onnx = outer_config.get("simplify_onnx", False)
        self.use_disj_adapter = outer_config.get("use_disj_adapter", True)

        self.instance_pre_filter_batch_size = outer_config.get(
            "instance_pre_filter_batch_size", None
        )
        self.instance_pre_filter_batch_size = (
            self.instance_pre_filter_batch_size
            if isinstance(self.instance_pre_filter_batch_size, int)
            else None
        )


class AbstractDomain(Enum):
    zono = "zono"
    box = "box"
    hbox = "hbox"
    dp = "dp"
    dpf = "DPF"
    none = "none"


class DomainSplittingConfig:
    initial_splits: int
    max_depth: int
    domain: AbstractDomain
    batch_size: int
    split_factor: int
    initial_split_dims: List[int]

    def __init__(self, config: Bunch):
        splitting_config = Bunch(config.get("domain_splitting", Bunch()))
        self.initial_splits = splitting_config.get("initial_splits", 3)
        self.max_depth = splitting_config.get("max_depth", 10)
        self.domain = AbstractDomain(splitting_config.get("domain", "dp"))
        self.batch_size = splitting_config.get("batch_size", 100)
        self.split_factor = splitting_config.get("split_factor", self.initial_splits)
        self.initial_split_dims = splitting_config.get("initial_split_dims", [0])


class MNBabVerifierConfig:
    optimizer: MNBabOptimizerConfig
    bab: BranchAndBoundConfig
    backsubstitution: BacksubstitutionConfig
    outer: OuterVerifierConfig
    domain_splitting: DomainSplittingConfig

    def __init__(self, config: Bunch):
        self.optimizer = MNBabOptimizerConfig(config)
        self.bab = BranchAndBoundConfig(config)
        self.backsubstitution = BacksubstitutionConfig(config)
        self.outer = OuterVerifierConfig(config)
        self.domain_splitting = DomainSplittingConfig(config)


def make_verifier_config(**config: Any) -> MNBabVerifierConfig:
    bunch = Bunch(**config)
    return MNBabVerifierConfig(bunch)  # TODO: check spelling of config options


class Dtype(Enum):
    float32 = "float32"
    float64 = "float64"


class Config:
    random_seed: int  # Random seed used for reproducibility
    timeout: int  # Per property time out after which verification terminates (unsuccessfully)
    experiment_name: str
    use_gpu: bool
    dtype: Dtype  # Single or double precision
    logger: LoggerConfig
    input_dim: Tuple[int, ...]  # Dimensionality of input data
    test_data_path: str  # Path to datasets
    benchmark_instances_path: str  # Path to benchmark instance list
    eps: float  # l-inf perturbation magnitude to verify against
    network: NetworkConfig
    verifier: MNBabVerifierConfig

    def __init__(self, config: Bunch):
        self.random_seed = config.random_seed
        self.timeout = config.timeout
        self.experiment_name = config.experiment_name
        self.use_gpu = config.use_gpu
        self.dtype = Dtype(config.get("dtype", "float32"))
        self.load_eager = config.get("load_eager", False)
        self.logger = LoggerConfig(config)
        self.input_dim = tuple(config.input_dim)
        self.test_data_path = config.get("test_data_path", "")
        self.benchmark_instances_path = config.get("benchmark_instances_path", "")
        self.eps = config.get("eps", 0)
        self.network = NetworkConfig(config)
        self.normalization_means = config.get("normalization_means",None)
        self.normalization_stds = config.get("normalization_stds", None)
        self.verifier = MNBabVerifierConfig(config)


def make_config(**config: Any) -> Config:
    bunch = Bunch(**config)
    return Config(bunch)  # TODO: check spelling of config options
