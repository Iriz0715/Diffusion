import os, pathlib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
from tensorflow.keras.layers import Input

def convert_and_save_quantized_model(input_model_path, signature_input_shape, train_images):
    model = load_model(input_model_path, custom_objects={'GroupNormalization': tfa.layers.GroupNormalization})
    model = Model(inputs=model.input, outputs=model.outputs[0])

    @tf.function(input_signature=[tf.TensorSpec(signature_input_shape, tf.float32)])
    def signature_fn(input_tensor):
        return model(input_tensor)
    
    concrete_func = signature_fn.get_concrete_function()
    
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    # Representative dataset:
    def representative_data_gen():
        if train_images is None:
            raise ValueError('You must provide train_images or a dataset for representative data.')
        for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(500):
            yield [tf.cast(input_value, tf.float32)]
    
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen

    # Force int8 ops only 
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    # Convert
    tflite_model = converter.convert()
    
    #save the quantized TFLite model
    tflite_file_path = pathlib.Path(input_model_path.replace('.h5', '_uint8.tflite'))
    tflite_file_path.write_bytes(tflite_model)


def convert_and_save_float_model(input_model_path, signature_input_shape):
    model = load_model(input_model_path, custom_objects={'GroupNormalization': tfa.layers.GroupNormalization})
    model = Model(inputs=model.input, outputs=model.outputs[0])
    
    # Create a signature function with the desired input shape & type
    @tf.function(input_signature=[tf.TensorSpec(signature_input_shape, tf.float32)])
    def signature_fn(input_tensor):
        return model(input_tensor)
    
    concrete_func = signature_fn.get_concrete_function()
    
    # Create TFLite converter from concrete function
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    # Optimization will actually cause the model to run slower and lead to slight differences
    # It will reduce the size of the model
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    #Convert
    tflite_model = converter.convert()
    
    # Save to output_filename (can adjust directory as needed)
    output_dir = os.path.dirname(input_model_path)
    tflite_file_path = pathlib.Path(input_model_path.replace('.h5', '_float32.tflite'))
    tflite_file_path.write_bytes(tflite_model)