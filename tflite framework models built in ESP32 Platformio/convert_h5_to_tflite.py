import tensorflow as tf

def convert_h5_to_tflite(h5_path, output_path):
    print(f"Loading model from {h5_path}...")
    model = tf.keras.models.load_model(h5_path, compile=False)
    print("Model loaded, converting...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    

    # 启用 SELECT_TF_OPS 以支持 TensorList 操作
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.allow_custom_ops = False
    # 禁用降低 TensorList 操作
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved to {output_path}, size: {len(tflite_model)} bytes")

# 输入路径
conv_h5 = r"D:\Oswaldo's surf project\DR O's database\models\conv1d_autoencoder_multi_modal100k_data.h5"


# 转换 Conv1D 模型
convert_h5_to_tflite(conv_h5, "conv_100k_model.tflite")

print("Done!")