import onnx  # type: ignore [import]

from dnnv.dnnv.nn.parser.onnx import _parse_onnx_model  # type: ignore [import]
from dnnv.dnnv.nn.transformers.simplifiers import simplify  # type: ignore [import]


def simplify_onnx(onnx_file: onnx.ModelProto) -> onnx.ModelProto:

    op_graph = _parse_onnx_model(onnx_file)
    simplified_op_graph1 = simplify(op_graph)
    # simplified_op_graph1.export_onnx("output.onnx")
    return simplified_op_graph1.as_onnx()
