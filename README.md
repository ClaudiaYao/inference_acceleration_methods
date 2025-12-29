# Inference Acceleration for Beginners

- transformer-torch-tensorrt.ipynb: use torch-tensorrt library to convert Pytorch model to TensorRT-optimized model
- transformer_pytorch_onnx.ipynb: use torch.export.onnx method to convert Pytorch model to onnx model, and use onnxruntime library to run onnx model.

## For tensorRT inference:

- prepare-onnx.py: Prerequisite to use tensorRT to do inference, generate an .onnx file.
- run-tensorrt.py: use tensorRT to do inference.
