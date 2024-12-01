import tensorflow as tf
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

# Paths to content and style images
content_path = "content.jpg"
style_path = "style.jpg"

# Helper functions
def load_and_process_image(image_path):
    max_dim = 512  # Resize for faster computation
    img = load_img(image_path)
    img = img_to_array(img)
    img = tf.image.resize(img, (max_dim, max_dim))
    img = np.expand_dims(img, axis=0)
    return vgg19.preprocess_input(img)

def deprocess_image(img):
    img = img[0]  # Remove batch dimension
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]  # Convert BGR to RGB
    img = np.clip(img, 0, 255).astype('uint8')
    return img

# Load images
content_image = load_and_process_image(content_path)
style_image = load_and_process_image(style_path)

# Define VGG19 model
vgg = vgg19.VGG19(include_top=False, weights="imagenet")
vgg.trainable = False

# Layers for style and content features
style_layers = [
    "block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"
]
content_layers = ["block5_conv2"]

# Extract features from layers
def get_model():
    outputs = [vgg.get_layer(name).output for name in style_layers + content_layers]
    return tf.keras.Model([vgg.input], outputs)

def gram_matrix(tensor):
    """Calculate the Gram matrix for style representation."""
    channels = int(tensor.shape[-1])
    tensor = tf.reshape(tensor, [-1, channels])
    gram = tf.matmul(tensor, tensor, transpose_a=True)
    return gram / tf.cast(tf.shape(tensor)[0], tf.float32)

def compute_loss(model, loss_weights, generated_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights
    model_outputs = model(generated_image)
    
    # Separate style and content features
    style_outputs = model_outputs[:len(style_layers)]
    content_outputs = model_outputs[len(style_layers):]
    
    # Style loss
    style_loss = tf.add_n([
        tf.reduce_mean((gram_matrix(style_output) - gram_matrix(target))**2)
        for style_output, target in zip(style_outputs, gram_style_features)
    ])
    style_loss *= style_weight / len(style_layers)
    
    # Content loss
    content_loss = tf.add_n([
        tf.reduce_mean((content_output - target)**2)
        for content_output, target in zip(content_outputs, content_features)
    ])
    content_loss *= content_weight / len(content_layers)
    
    total_loss = style_loss + content_loss
    return total_loss

def run_style_transfer(content_path, style_path, iterations=1000, style_weight=1e-2, content_weight=1e4):
    model = get_model()
    for layer in model.layers:
        layer.trainable = False
    
    content_image = load_and_process_image(content_path)
    style_image = load_and_process_image(style_path)
    
    # Extract features
    style_features = model(style_image)[:len(style_layers)]
    content_features = model(content_image)[len(style_layers):]
    gram_style_features = [gram_matrix(feature) for feature in style_features]
    
    # Generate initial image
    generated_image = tf.Variable(content_image, dtype=tf.float32)
    
    # Optimizer
    opt = tf.optimizers.Adam(learning_rate=5.0)
    
    # Loss weights
    loss_weights = (style_weight, content_weight)
    
    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            total_loss = compute_loss(model, loss_weights, generated_image, gram_style_features, content_features)
        grads = tape.gradient(total_loss, generated_image)
        opt.apply_gradients([(grads, generated_image)])
        return total_loss
    
    for i in range(iterations):
        loss = train_step()
        if i % 100 == 0:
            print(f"Iteration {i}: Loss: {loss.numpy()}")
    
    final_image = deprocess_image(generated_image.numpy())
    return final_image

# Run style transfer
output_image = run_style_transfer(content_path, style_path)

# Save and display result
plt.imshow(output_image)
plt.axis("off")
plt.show()
