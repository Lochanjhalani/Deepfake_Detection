import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from preprocess import preprocess_image

# Load trained model
model = tf.keras.models.load_model("../models/deepseek_model.h5")

def get_last_conv_layer(model):
    """
    Find the last convolutional layer in the model.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name  # Return last convolutional layer name
    raise ValueError("No convolutional layer found in model.")

def generate_gradcam(image_path):
    """
    Generates Grad-CAM heatmap for the given image.
    """
    # Preprocess image
    img = preprocess_image(image_path)

    # Get last convolutional layer name dynamically
    last_conv_layer_name = get_last_conv_layer(model)

    # Create a model that maps input image to last conv layer and output predictions
    grad_model = tf.keras.Model(
        [model.input], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute gradient of top predicted class with respect to feature maps
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        top_class = tf.argmax(predictions[0])
        loss = predictions[:, top_class]

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Apply Grad-CAM
    conv_outputs = conv_outputs[0]  # Remove batch dimension
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1  # Avoid division by zero

    # Read original image
    img_orig = cv2.imread(image_path)
    img_orig = cv2.resize(img_orig, (224, 224))  # Resize to match model input

    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap.numpy(), (img_orig.shape[1], img_orig.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose heatmap onto original image
    superimposed_img = cv2.addWeighted(img_orig, 0.7, heatmap, 0.3, 0)

    # Display the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM Heatmap")
    plt.show()

if __name__ == "__main__":
    generate_gradcam("../static/sample.jpg")
