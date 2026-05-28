import onnxslim

inputpath  = r'../inference/rec_onnx/best_u8.onnx'
outputpath = r'../inference/rec_onnx/best-smi.onnx'

onnxslim.slim(inputpath, outputpath)
print('ok!!!')

# ── 旧方案（onnxsim + onnxoptimizer）──────────────────────────────
# import onnx
# from onnxsim import simplify
# import onnxoptimizer
#
# model = onnx.load(inputpath)
# newmodel = onnxoptimizer.optimize(model)
# model_simp, check = simplify(newmodel)
# assert check, "Simplified ONNX model could not be validated"
# onnx.save(model_simp, outputpath)
# print('ok!!!')
