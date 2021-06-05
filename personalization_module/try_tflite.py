import numpy as np
import tensorflow as tf

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="C:/Users/anand/Documents/Current/DeepTricorder/personalization_module/model.tflite")

# Get input and output tensors.
print("Before reshaping: ")
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("input details: " + str(input_details))
print("output details: " + str(output_details))

# resize input for 10 frames
interpreter.resize_tensor_input(input_details[0]['index'], (20, 36, 36, 3))
interpreter.allocate_tensors()

print("After reshaping: ")
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("input details: " + str(input_details))
print("output details: " + str(output_details))

