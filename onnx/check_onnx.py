# -*- coding: utf-8 -*-
"""
Time:     2023/12/6 15:17
Author:   cjn
Version:  1.0.0
File:     pth2onnx.py
Describe:
"""
import onnxruntime
from onnxruntime.datasets import get_example
import torch

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

dummy_input = torch.randn(1, 3, 224, 224, device='cpu')

onnx_model = get_example("/home/hzbz/PycharmProjects/image-classification-pytorch/model.onnx")
sess = onnxruntime.InferenceSession(onnx_model)

onnx_result = sess.run(None, {'input': to_numpy(dummy_input)})
print(onnx_result)

model_path = '/home/hzbz/tool/rknn-toolkit2-master/cjn/bestmodel.pth'
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()
pytorch_result = model(dummy_input)
print(pytorch_result)

