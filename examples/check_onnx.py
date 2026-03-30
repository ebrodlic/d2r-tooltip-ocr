import onnx

model = onnx.load("models/d2r_tooltip_crnn_best.onnx")
for input in model.graph.input:
    dims = [d.dim_value for d in input.type.tensor_type.shape.dim]
    print(dims)