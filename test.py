import onnx

def test_onnx():
    try:
        # 创建一个简单的ONNX模型
        import onnx.helper as helper
        node = helper.make_node("Add", ["x", "y"], ["z"])
        graph = helper.make_graph(
            [node],
            "test-model",
            [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, (1,)),
             helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info("z", onnx.TensorProto.FLOAT, (1,))],
        )
        model = helper.make_model(graph)
        model_bytes = model.SerializeToString()
        
        # 尝试加载模型
        loaded_model = onnx.load_model_from_string(model_bytes)
        print("ONNX model loaded successfully.")
    except AttributeError as e:
        print(f"ONNX attribute error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    test_onnx()
