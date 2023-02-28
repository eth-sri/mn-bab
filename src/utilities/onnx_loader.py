import warnings
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

import numpy as np
import onnx  # type: ignore[import]
import torch
from onnx import numpy_helper
from torch import Tensor, nn

from onnx2pytorch.onnx2pytorch.convert.operations import (  # type: ignore[import] # noqa: E402; noqa: E402
    Loop,
    convert_operations,
    get_buffer_name,
)
from onnx2pytorch.onnx2pytorch.operations.div import Div  # type: ignore[import]
from onnx2pytorch.onnx2pytorch.operations.mul import Mul
from onnx2pytorch.onnx2pytorch.operations.reducesum import (
    ReduceSum,  # type: ignore[import]
)
from onnx2pytorch.onnx2pytorch.operations.shape import Shape  # type: ignore[import]
from onnx2pytorch.onnx2pytorch.operations.split import Split  # type: ignore[import]
from onnx2pytorch.onnx2pytorch.utils import (  # type: ignore[import]  # noqa: E402
    get_inputs_names,
    get_outputs_names,
)
from src.concrete_layers.binary_op import BinaryOp
from src.concrete_layers.concat import Concat
from src.concrete_layers.multi_path_block import MultiPathBlock
from src.concrete_layers.pad import Pad
from src.concrete_layers.permute import Permute
from src.concrete_layers.reshape import Reshape
from src.concrete_layers.residual_block import ResidualBlock
from src.concrete_layers.slice import Slice
from src.concrete_layers.split_block import SplitBlock
from src.concrete_layers.unbinary_op import UnbinaryOp


class ConvertModel(nn.Module):
    def __init__(
        self,
        onnx_model: onnx.ModelProto,
        batch_dim: int = 0,
        experimental: bool = False,
        debug: bool = False,
        enable_pruning: bool = False,
    ):
        """
        Convert onnx model to pytorch.

        Parameters
        ----------
        onnx_model: onnx.ModelProto
            Loaded onnx model.
        batch_dim: int
            Dimension of the batch.
        experimental: bool
            Experimental implementation allows batch_size > 1. However,
            batchnorm layers could potentially produce false outputs.
        enable_pruning: bool
            Track kept/pruned indices between different calls to forward pass.

        Returns
        -------
        model: torch.nn.Module
            A converted pytorch model.
        """
        super().__init__()
        self.onnx_model = onnx_model
        self.batch_dim = batch_dim
        self.experimental = experimental
        self.debug = debug
        self.enable_pruning = enable_pruning

        self.input_names = get_inputs_names(onnx_model.graph)
        self.output_names = get_outputs_names(onnx_model.graph)
        opset_version = onnx_model.opset_import[0].version

        # Store initializers as buffers
        buffer_names = set({})
        inits: Dict[str, Tensor] = {}
        for tensor in self.onnx_model.graph.initializer:
            buffer_name = get_buffer_name(tensor.name)
            buffer_names.add(tensor.name)
            self.register_buffer(
                buffer_name,
                torch.from_numpy(numpy_helper.to_array(tensor)),
            )
            inits[tensor.name] = torch.from_numpy(numpy_helper.to_array(tensor))

        # TODO Set values for Constant nodes as init values
        const_nodes = [n for n in self.onnx_model.graph.node if n.op_type == "Constant"]

        self.const_nodes: Dict[str, Any] = {}
        for cn in const_nodes:
            self.const_nodes[cn.output[0]] = torch.from_numpy(
                numpy_helper.to_array(cn.attribute[0].t)
            )

        self.all_consts: Dict[str, Any] = {**inits, **self.const_nodes}

        # Create mapping from node (identified by first output) to submodule
        self.mapping: Dict[str, str] = {}
        for op_id, op_name, op in convert_operations(
            onnx_model.graph, opset_version, batch_dim, enable_pruning, self.all_consts
        ):
            setattr(self, op_name, op)
            if isinstance(op, Loop) and debug:
                raise NotImplementedError("debug-mode with Loop node not implemented.")
            self.mapping[op_id] = op_name

        # Compute activation dependencies, mapping each node to its dependents
        self.needed_by = compute_activation_dependencies(
            self.onnx_model.graph, self, self.mapping
        )

        # Reverse activation dependencies to identify merge nodes for ResNets
        self.cg_needs: Dict[str, List[str]] = {}  # Pure computation graph
        self.needs: Dict[str, List[str]] = {}  # Contains initializers and constants
        for (k, v) in self.needed_by.items():
            for v_i in v:
                if (
                    "weight" not in k
                    and "bias" not in k
                    and k not in buffer_names
                    and k not in self.const_nodes
                ):
                    if v_i not in self.cg_needs:
                        self.cg_needs[v_i] = [k]
                    else:
                        self.cg_needs[v_i].append(k)
                if v_i not in self.needs:
                    self.needs[v_i] = [k]
                else:
                    self.needs[v_i].append(k)

        self.ordered_nodes = list(self.onnx_model.graph.node)

        if experimental:
            warnings.warn(
                "Using experimental implementation that allows 'batch_size > 1'."
                "Batchnorm layers could potentially produce false outputs."
            )

    def forward_trace_to_graph(self) -> nn.Sequential:

        id = self.input_names[0]
        module_list: List[nn.Module] = []
        from_multipath = False
        while id in self.needed_by:

            if len(list(self.needed_by[id])) > 1:
                merge_op = self.get_op_by_out_id(self._find_merge_id(id))
                if (
                    isinstance(merge_op, BinaryOp)
                    and merge_op.op == "add"
                    and len(list(self.needed_by[id])) == 2
                ):
                    # ResBlock suffices
                    res_block, out_id = self._build_res_block(id)
                    module_list.append(res_block)
                    from_multipath = False
                else:
                    multi_path, out_id = self._build_multi_path_block(id)
                    module_list.append(multi_path)
                    from_multipath = True
            else:
                if not from_multipath:  # Multipaths return the next element already
                    id = list(self.needed_by[id])[0]
                module, out_id = self._build_seq_model(
                    out_op_id=id, add_final_node=True
                )
                assert module is not None
                module_list.append(module)
                from_multipath = False
            id = out_id

        module = self._flatten_module_list(module_list)
        assert module is not None
        return module

    def forward_trace_to_graph_unet(self) -> nn.Sequential:

        # Specific parsing for the carvana-unet benchmark
        id = self.input_names[0]
        module_list: List[nn.Module] = []
        while id in self.needed_by:
            id = sorted(list(self.needed_by[id]), key=(lambda x: int(x)))[
                0
            ]  # We only follow the "nearest child"
            module, out_id = self._build_seq_model(
                out_op_id=id, add_final_node=True, parse_unet=True
            )
            assert module is not None
            module_list.append(module)
            id = out_id

        module = self._flatten_module_list(module_list)
        assert module is not None
        return module

    def _flatten_module_list(self, module_list: List[nn.Module]) -> nn.Sequential:
        """Only does a shallow flatten pass"""

        flat_mod_list: List[nn.Module] = []
        if len(module_list) == 1 and isinstance(module_list[0], nn.Sequential):
            return module_list[0]

        for m in module_list:
            if isinstance(m, nn.Sequential):
                flat_mod_list.extend(m)
            else:
                flat_mod_list.append(m)  # type:ignore [arg-type]

        return nn.Sequential(*flat_mod_list)

    def _build_seq_model(
        self,
        out_op_id: str,
        convert_to_seq: bool = True,
        add_final_node: bool = False,
        parse_unet: bool = False,
    ) -> Tuple[Optional[nn.Sequential], str]:
        seq_layers = []

        while True:
            op = self.get_op_by_out_id(out_op_id)

            if (
                len(self.cg_needs[out_op_id]) > 1
            ):  # Found merge node at end of sequential path
                break

            # Special NN4Sys Split-Block (Note: only has self.needed_by[out_op_id]==1)
            if isinstance(op, Split):
                op, out_op_id = self._build_split_block(out_op_id)

            if out_op_id in self.needed_by and len(self.needed_by[out_op_id]) > 1:
                if len(self.needed_by[out_op_id]) == 2:  # Default residual block
                    seq_layers.append(
                        op
                    )  # Add the header as residual nets don't carry it
                    op, out_op_id = self._build_res_block(out_op_id)
                else:  # Multipath residual block
                    op, out_op_id = self._build_multi_path_block(out_op_id, parse_unet)
            else:
                out_op_id = list(self.needed_by[out_op_id])[0]

            if op is not None:
                seq_layers.append(op)

            if (
                out_op_id not in self.needed_by and add_final_node
            ):  # The final node in the graph
                op = self.get_op_by_out_id(out_op_id)
                if op is not None:
                    seq_layers.append(op)
                break

        if convert_to_seq:
            seq_layers = [
                layer for layer in seq_layers if issubclass(type(layer), nn.Module)
            ]  # Filters out Argmax
            return nn.Sequential(*seq_layers), out_op_id
        else:
            return None, out_op_id

    def _build_multi_path_block(
        self, out_op_id: str, parse_unet: bool = False
    ) -> Tuple[nn.Module, str]:

        needed_by_id_list = list(self.needed_by[out_op_id])
        bb: nn.Module

        # Handle flatten
        if len(needed_by_id_list) == 2:
            is_flatten, flatten_merge_id = self._is_flatten(
                needed_by_id_list[0], needed_by_id_list[1]
            )
            if is_flatten:
                bb = nn.Flatten()
                merge_id = flatten_merge_id

                next_id_list = list(self.needed_by[merge_id])
                assert len(next_id_list) == 1
                ret_id = next_id_list[0]

                return bb, ret_id

        try:
            header = self.get_op_by_out_id(out_op_id)
        except KeyError:
            header = None

        merge_ids: List[str] = []
        paths: List[nn.Sequential] = []
        sorted_ids = sorted(list(needed_by_id_list), key=(lambda x: int(x)))
        if parse_unet:
            sorted_ids = [
                sorted_ids[0],
                sorted_ids[-1],
            ]  # Only follow lowest and highest path

        for path_start_id in sorted_ids:

            if parse_unet:
                # Carvana-Unet Upsample specific - We want to skip all padding calculations
                op = self.get_op_by_out_id(out_op_id)
                if isinstance(op, nn.ConvTranspose2d):
                    pad_id = sorted_ids[-1]
                    pad_op = self.get_op_by_out_id(pad_id)
                    assert isinstance(pad_op, Pad)
                    pad_op.pad = (0, 1, 0, 1)
                    unet_merge_id = list(self.needed_by[pad_id])[0]
                    return nn.Sequential(*[op, pad_op]), unet_merge_id

                path, merge_id = self._build_seq_model(
                    path_start_id,
                    convert_to_seq=True,
                    add_final_node=False,
                    parse_unet=parse_unet,
                )
                if path is None:
                    path = nn.Sequential(*[nn.Identity()])

                paths.append(path)
                merge_ids.append(merge_id)
            else:
                path, merge_id = self._build_seq_model(path_start_id)
                assert path is not None
                paths.append(path)
                merge_ids.append(merge_id)

        merge_id = merge_ids[0]
        for i, mi in enumerate(merge_ids):
            assert mi == merge_id
            assert isinstance(paths[i], nn.Sequential)

        merge = self.get_op_by_out_id(merge_id)

        if parse_unet:  # Very dirty u-net hack
            if len(paths[1]) == 0:
                paths.reverse()

        if header is not None and not isinstance(header, Slice):
            bb = nn.Sequential(
                *(header, MultiPathBlock(header=None, paths=paths, merge=merge))
            )
        else:
            bb = MultiPathBlock(header=header, paths=paths, merge=merge)

        # Move over the merge_block
        if merge_id not in self.needed_by:  # We are at the final node
            return bb, merge_id
        next_id_list = list(self.needed_by[merge_id])
        assert len(next_id_list) == 1
        ret_id = next_id_list[0]

        return bb, ret_id

    def _is_flatten(
        self, path_a_start_id: str, path_b_start_id: str
    ) -> Tuple[bool, str]:
        is_a_flatten = isinstance(self.get_op_by_out_id(path_a_start_id), Shape)
        is_b_flatten = isinstance(self.get_op_by_out_id(path_b_start_id), Shape)
        if is_a_flatten or is_b_flatten:  # Actually we run a flatten operation
            if is_a_flatten:
                _, merge_id_a = self._build_seq_model(
                    path_a_start_id, convert_to_seq=False
                )
                assert (
                    merge_id_a == path_b_start_id
                ), "Unknown sequence of shape operators"
                merge_id = merge_id_a
            else:
                _, merge_id_b = self._build_seq_model(
                    path_b_start_id, convert_to_seq=False
                )
                assert (
                    merge_id_b == path_a_start_id
                ), "Unknown sequence of shape operators"
                merge_id = merge_id_b
            return (True, merge_id)
        else:
            return (False, path_a_start_id)

    def _build_res_block(self, out_op_id: str) -> Tuple[nn.Module, str]:

        assert (
            len(self.needed_by[out_op_id]) == 2
        ), "Cannot deal with more than 2 paths in  ResidualBlock"
        needed_by_id_list = list(self.needed_by[out_op_id])
        path_a_start_id, path_b_start_id = needed_by_id_list[0], needed_by_id_list[1]

        # Handle Flatten via Res-Block
        is_flatten, flatten_merge_id = self._is_flatten(
            path_a_start_id, path_b_start_id
        )
        bb: nn.Module
        if is_flatten:  # Actually we run a flatten operation
            merge_id = flatten_merge_id
            bb = nn.Flatten()
        else:
            path_a, merge_id_a = self._build_seq_model(
                path_a_start_id,
            )
            path_b, merge_id_b = self._build_seq_model(
                path_b_start_id,
            )
            assert merge_id_a == merge_id_b, "Paths try to merge at different points"
            merge_id = merge_id_a
            assert isinstance(path_a, nn.Sequential)
            assert isinstance(path_b, nn.Sequential)
            bb = ResidualBlock(path_a=path_a, path_b=path_b)
        return bb, merge_id

    def _build_split_block(self, out_op_id: str) -> Tuple[nn.Module, str]:

        split_center_id = out_op_id
        split_res_id = str(int(out_op_id) + 1)  # :/

        assert (
            len(self.needed_by[split_center_id]) == 1
        ), "Center path split needed only once."
        center_start_id = list(self.needed_by[split_center_id])[0]
        assert len(self.needed_by[split_res_id]) == 2, "Expected 2 residual paths."

        center_path, inner_merge_id = self._build_seq_model(
            center_start_id,
        )

        # Check inner merge
        assert (
            inner_merge_id in self.needed_by[split_res_id]
        ), "Unkown inner merge point"
        assert isinstance(self.get_op_by_out_id(inner_merge_id), Mul)
        # Handle reduce sums
        outer_reduce_id = list(self.needed_by[split_res_id] - {inner_merge_id})[0]
        inner_reduce_id = list(self.needed_by[inner_merge_id])[0]
        assert isinstance(self.get_op_by_out_id(inner_reduce_id), ReduceSum)
        assert isinstance(self.get_op_by_out_id(outer_reduce_id), ReduceSum)

        # Check final div
        inner_red_merge_id = list(self.needed_by[inner_reduce_id])[0]
        outer_red_merge_id = list(self.needed_by[outer_reduce_id])[0]
        assert inner_red_merge_id == outer_red_merge_id, "No unique final merge point"
        merge_id = inner_red_merge_id
        assert isinstance(
            self.get_op_by_out_id(merge_id), Div
        ), "Not div as final merge"

        # Split-Info
        split = self.get_op_by_out_id(split_center_id)
        split_info: Tuple[bool, Tuple[int, ...], Optional[int], int, bool] = (
            split.enable_pruning,  # type:ignore[assignment]
            split.split_size_or_sections,  # type:ignore[assignment]
            split.number_of_splits,  # type:ignore[assignment]
            split.dim,  # type:ignore[assignment]
            split.keep_size,  # type:ignore[assignment]
        )
        # Reduce-Info
        inner_reduce = self.get_op_by_out_id(inner_reduce_id)
        outer_reduce = self.get_op_by_out_id(outer_reduce_id)
        red_inner_info: Tuple[int, bool, bool] = (
            inner_reduce.dim,  # type:ignore[assignment]
            inner_reduce.keepdim,  # type:ignore[assignment]
            inner_reduce.noop_with_empty_axes,  # type:ignore[assignment]
        )
        red_outer_info: Tuple[int, bool, bool] = (
            outer_reduce.dim,  # type:ignore[assignment]
            outer_reduce.keepdim,  # type:ignore[assignment]
            outer_reduce.noop_with_empty_axes,  # type:ignore[assignment]
        )
        assert center_path is not None
        return (
            SplitBlock(split_info, center_path, red_inner_info, red_outer_info),
            merge_id,
        )

    def _find_merge_id(self, start_id: str) -> str:

        needed_by_id_list = list(self.needed_by[start_id])
        merge_ids: List[str] = []

        def sort_fun(x: str) -> int:
            try:
                return int(x)
            except Exception:
                return int("".join(x for x in x if x.isdigit()))

        sorted_ids = sorted(list(needed_by_id_list), key=sort_fun)

        for path_start_id in sorted_ids:
            _, merge_id = self._build_seq_model(path_start_id)
            merge_ids.append(merge_id)

        merge_id = merge_ids[0]
        for mi in merge_ids:
            assert mi == merge_id
        return merge_id

    def get_op_by_out_id(self, out_op_id: str) -> nn.Module:

        out_op_name = self.mapping[out_op_id]
        out_op_name_pref = out_op_name.split("_")[0].lower()
        op = getattr(self, out_op_name)

        # @Fix the operator order for Linear in case the MatMul is with an A argument
        if out_op_name_pref == "matmul":
            if (
                self.cg_needs[out_op_id][0] == self.needs[out_op_id][1]
            ):  # Multiply with matrix rhs A*x
                has_bias = op.bias is not None
                new_ll = nn.Linear(
                    in_features=op.out_features,
                    out_features=op.in_features,
                    bias=has_bias,
                )
                new_ll.weight = nn.Parameter(op.weight.detach().T)
                assert op.bias.dim() == 1
                new_ll.bias = nn.Parameter(op.bias.detach())
                # new_ll.bias = nn.Parameter(op.bias.detach().T)
                op = new_ll
            return op

        # Deals with operations where one part is already pre-loaded
        if (
            out_op_name_pref in {"add", "sub", "div", "mul"}
            and len(self.cg_needs[out_op_id]) <= 1
        ):  # Ignore Add's at merge point
            assert len(self.needs[out_op_id]) == 2, "Too many inputs for Binary Op"
            val = None
            apply_right = False
            if self.needs[out_op_id][0] in self.all_consts:
                val = self.all_consts[self.needs[out_op_id][0]]
                apply_right = True
            else:
                assert (
                    self.needs[out_op_id][1] in self.all_consts
                ), "Value not not found for UnbinaryOp"
                val = self.all_consts[self.needs[out_op_id][1]]

            return UnbinaryOp(out_op_name_pref, const_val=val, apply_right=apply_right)

        # Deals with batchnorm
        if out_op_name_pref == "batchnormalization":
            bnu = op.bnu
            new_bn = nn.BatchNorm2d(
                num_features=bnu.num_features, eps=bnu.eps, momentum=bnu.momentum
            )
            new_bn.running_mean = bnu.running_mean
            new_bn.running_var = bnu.running_var
            new_bn.weight = bnu.weight
            new_bn.bias = bnu.bias
            return new_bn

        # Deals with Padding layers
        if out_op_name_pref == "pad":
            return Pad(op.padding, op.mode, op.constant)

        # Deals with Transpose/Permutation layers - # TODO not totally clean
        if out_op_name_pref == "transpose":
            return Permute(op.dims)

        if out_op_name_pref == "reshape":
            if (
                op.shape is None and len(self.needs[out_op_id]) > 1
            ):  # Actually load shape from constant
                assert len(self.needs[out_op_id]) == 2
                if self.needs[out_op_id][0] in self.const_nodes:
                    op.shape = self.const_nodes[self.needs[out_op_id][0]].numpy()
                if self.needs[out_op_id][1] in self.const_nodes:
                    op.shape = self.const_nodes[self.needs[out_op_id][1]].numpy()
            if op.shape is None or len(op.shape) == 1:
                return nn.Flatten()
            elif len(op.shape) == 2 and np.prod(op.shape) < 0:  # -1 as dimension
                return nn.Flatten()
            else:
                # Assume that first dim is batch-size
                return Reshape(list(op.shape)[1:])  # type: ignore[arg-type]

        if out_op_name_pref == "slice":
            op = Slice(
                dim=op.dim[0], starts=op.starts[0], ends=op.ends[0], steps=op.steps[0]
            )

        if out_op_name_pref == "concat":
            op = Concat(dim=op.dim)

        if out_op_name_pref in {"add", "sub"} and len(self.cg_needs[out_op_id]) == 2:
            assert len(self.needs[out_op_id]) == 2, "Too many inputs for Binary Op"
            return BinaryOp(out_op_name_pref)
        return op


def compute_activation_dependencies(
    onnx_graph: Any, model: ConvertModel, mapping: Dict[str, str]
) -> DefaultDict[Any, Set[Any]]:
    """
    Compute activation dependencies, mapping each node to its dependents.

    Parameters
    ----------
    onnx_graph: onnx.GraphProto
        ONNX graph.
    model: onnx2pytorch.ConvertModel
        Module which contains converted submodules.
    mapping: dict
        Dictionary mapping from node name to name of submodule.

    Returns
    -------
    needed_by: dict
        Dictionary mapping from node name to names of its dependents.
    """
    needed_by: DefaultDict[Any, Set[Any]] = defaultdict(set)
    for node in onnx_graph.node:
        out_op_id = node.output[0]
        for in_op_id in node.input:
            needed_by[in_op_id].add(out_op_id)
        if node.op_type == "Loop":
            # Look at nodes in the loop body
            l1 = getattr(model, mapping[out_op_id])  # Loop object
            loop_body_l1 = l1.body
            for node_l1 in loop_body_l1.node:
                for in_op_id in node_l1.input:
                    # Treating node (outer loop) as dependent, not node_l1
                    needed_by[in_op_id].add(out_op_id)
                if node_l1.op_type == "Loop":
                    # Look at nodes in the loop body
                    l2 = getattr(model, l1.mapping[node_l1.output[0]])  # Loop object
                    loop_body_l2 = l2.body
                    for node_l2 in loop_body_l2.node:
                        for in_op_id in node_l2.input:
                            # Treating node (outer loop) as dependent, not node_l2
                            needed_by[in_op_id].add(out_op_id)
                        if node_l2.op_type == "Loop":
                            # TODO: make this recursive for nested loops
                            raise NotImplementedError(
                                "Activation garbage collection not implemented for >2 nested loops."
                            )
    needed_by.default_factory = None
    return needed_by
