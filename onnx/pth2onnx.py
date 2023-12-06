# -*- coding: utf-8 -*-
"""
Time:     2023/12/6 14:50
Author:   cjn
Version:  1.0.0
File:     pth2onnx.py
Describe: 
"""
import torch.onnx

model_path = ""

model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

input_names = ['input']
output_names = ['output']

x = torch.randn(1, 3, 224, 224, requires_grad=True)
torch.onnx.export(model, x, 'model.onnx', verbose=True, input_names=input_names, output_names=output_names)
