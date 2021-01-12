import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

from torchvision import models
import cv2
import torch
from torch import nn
from albumentations import Resize, Compose
from albumentations.pytorch.transforms import  ToTensor
from albumentations.augmentations.transforms import Normalize
from networks.pointpillars8_shared import RPN

import onnx
import torchvision.transforms as transforms
from PIL import Image



# onnx_file_path = 'resnet50.onnx'
# engine_path = 'resnet50.engine'

onnx_file_path = 'rpn.onnx'
engine_path = 'rpn.engine'


def export():
	# model = models.resnet50(pretrained=True)
	# input = torch.randn(1, 3, 224, 224, device='cuda')
	# input = preprocess_image("turkish_coffee.jpg").cuda()
	
	model = RPN(64)
	input = torch.randn(1, 64, 800, 800, device='cuda')
	model.eval().cuda()
        
	# torch.onnx.export(model, input, onnx_file_path, verbose=False, input_names=['input'], output_names=['output'], export_params=True, opset_version=12, dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})
	
	torch.onnx.export(model, input, onnx_file_path, verbose=False, input_names=['input'], output_names=['output'], export_params=True)# , opset_version=12 
	
	onnx_model = onnx.load(onnx_file_path)
	onnx.checker.check_model(onnx_model)
	onnx.helper.printable_graph(onnx_model.graph)
	
  
def build_engine(engine_path):
	TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
	EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
	with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
		# builder.max_workspace_size = 2**30 # 1 << 30
		builder.max_batch_size = 1 # always 1 for explicit batch
		with open(onnx_file_path, 'rb') as model:
			parser.parse(model.read())
		
		config = builder.create_builder_config()
		config.max_workspace_size = 1 << 30
		config.flags = 1 << (int)(trt.BuilderFlag.FP16)
		## profile = builder.create_optimization_profile();
		## profile.set_shape("input", (1, 15, 9), (16000, 15, 9), (16000, 15, 9)) 
		## profile.set_shape("input", (1, 3, 224, 224), (10, 3, 224, 224), (10, 3, 224, 224)) 
		## config.add_optimization_profile(profile)
            
		print(network.num_layers)
		print(network.num_inputs)
		print(network.num_outputs)
		print("building engine")
		engine = builder.build_engine(network, config)
		print("completed")
		# return builder.build_cuda_engine(network)
		with open(engine_path, 'wb') as f:
			f.write(engine.serialize())


def infer(context, h_input, data_type=np.float32):		
	# dynamic shape, batch size
	## context.active_optimization_profile = 0
	## context.set_binding_shape(0, (h_input.shape))
	# print(context.get_binding_shape(0), context.get_binding_shape(1))
	size = trt.volume(h_input.shape) * np.dtype(data_type).itemsize 
	d_input = cuda.mem_alloc(size)
	h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=data_type)
	d_output = cuda.mem_alloc(h_output.nbytes)

	stream = cuda.Stream()
	cuda.memcpy_htod_async(d_input, h_input, stream)
	context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
	cuda.memcpy_dtoh_async(h_outpdataut, d_output, stream)
	stream.synchronize()
	
	return h_output
	
import time
import random
if __name__ == '__main__':
	'''
	export()
	
	build_engine(engine_path)
	'''
	print("loading engine")
	TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
	with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
		engine = runtime.deserialize_cuda_engine(f.read())
	
	context = engine.create_execution_context()

	num_loops = 100
	elapse = 0.0
	h_input = np.array(np.random.randn(1, 64, 800, 800), dtype=np.float32, order='C')
	data_type = np.float32
	size = trt.volume(h_input.shape) * np.dtype(data_type).itemsize 
	d_input = cuda.mem_alloc(size)
	h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=data_type)
	d_output = cuda.mem_alloc(h_output.nbytes)
		
	for i in range(num_loops):
		
		# host_output = infer(context, h_input, data_type=np.float32)
		
		stream = cuda.Stream()
		cuda.memcpy_htod_async(d_input, h_input, stream)
		stream.synchronize()
		start = time.time()
		context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
		stream.synchronize()
		elapse += time.time() - start
		cuda.memcpy_dtoh_async(h_output, d_output, stream)
		stream.synchronize()
	print(elapse / num_loops * 1000, 'ms')
	
	# output_data = torch.Tensor(host_output) # .reshape(engine.max_batch_size, -1)
	# h_input = np.array(np.random.randn(10, 3, 224, 224), dtype=np.float32, order='C')
	# h_input = np.array(preprocess_image("turkish_coffee.jpg").numpy(), dtype=np.float32, order='C')
	# host_output = infer(context, h_input, data_type=np.float32)
	# output_data = torch.Tensor(host_output).reshape(1, -1)
	# postprocess(output_data)
