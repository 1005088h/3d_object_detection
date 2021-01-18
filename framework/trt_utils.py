import onnx
import tensorrt as trt

def export_onnx(model, input, onnx_file_path, dynamic=False, input_names=['input'], output_names=['output']):
    if dynamic:
        torch.onnx.export(model, input, onnx_file_path, verbose=False, input_names=['input'], output_names=['output'],
                          opset_version=11, dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    else:
        torch.onnx.export(model, input, onnx_file_path, verbose=False, opset_version=11, input_names=input_names,
                          output_names=output_names)

    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)
    onnx.helper.printable_graph(onnx_model.graph)


def build_engine(onnx_file_path, engine_path, dynamic=False):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network,
                                                                                                               TRT_LOGGER) as parser:
        builder.max_batch_size = 1  # always 1 for explicit batch
        with open(onnx_file_path, 'rb') as model:
            parser.parse(model.read())

        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30
        config.flags = 1 << (int)(trt.BuilderFlag.FP16)
        if dynamic:
            profile = builder.create_optimization_profile()
            profile.set_shape("input", (1, 15, 9), (16000, 15, 9), (16000, 15, 9))
            config.add_optimization_profile(profile)
        print(network.num_layers)
        engine = builder.build_engine(network, config)
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        return engine


def load_engine(trt_engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(trt_engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine
