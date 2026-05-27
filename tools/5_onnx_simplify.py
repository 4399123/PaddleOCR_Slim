import onnx
from onnxsim import simplify
import onnxoptimizer

inputpath=r'../inference/rec_onnx/best_rename.onnx'
outputpath=r'../inference/rec_onnx/best-smi.onnx'

model = onnx.load(inputpath)
newmodel=onnxoptimizer.optimize(model)
model_simp, check = simplify(newmodel)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp,outputpath)
print('ok!!!')