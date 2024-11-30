import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf

def compute_loss(model, loss_weights, generated_image, gram_style_features, content_features):
    """
    Compute the style, content, and total variation loss.

    Args:
    - model: The pre-trained VGG19 model.
    - loss_weights: Tuple with weights for content and style losses.
    - generated_image: The generated image to optimize.
    - gram_style_features: Precomputed Gram matrices for style layers.
    - content_features: Precomputed content features.

    Returns:
    - total_loss: The combined loss.
    - (style_loss, content_loss, total_variation_loss): Individual components of the loss.
    """
    content_weight, style_weight = loss_weights
    
    # Extract features from the generated image
    model_outputs = model(generated_image)
    generated_content_features = model_outputs[:len(content_features)]
    generated_style_features = model_outputs[len(content_features):]
    
    # Compute content loss
    content_loss = tf.add_n(
        [tf.reduce_mean(tf.square(content_feature - gen_content_feature))
         for content_feature, gen_content_feature in zip(content_features, generated_content_features)]
    )
    
    # Compute style loss
    style_loss = tf.add_n(
        [tf.reduce_mean(tf.square(gram_style_feature - compute_gram_matrix(gen_style_feature)))
         for gram_style_feature, gen_style_feature in zip(gram_style_features, generated_style_features)]
    )
    
    # Scale losses
    content_loss *= content_weight
    style_loss *= style_weight

    # Total variation loss for smoothness
    total_variation_loss = tf.image.total_variation(generated_image)
    
    # Combine all losses
    total_loss = content_loss + style_loss + total_variation_loss
    return total_loss, (style_loss, content_loss, total_variation_loss)


def compute_gram_matrix(input_tensor):
    """
    Compute the Gram matrix for a given input tensor.

    Args:
    - input_tensor: A tensor from the style layers.

    Returns:
    - gram_matrix: The Gram matrix for the input tensor.
    """
    # Flatten the feature map along the spatial dimensions
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    
    # Compute the Gram matrix
    gram_matrix = tf.matmul(a, a, transpose_a=True)
    return gram_matrix / tf.cast(n, tf.float32)


def load_and_process_image(image_path, target_size=(512, 512)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def deprocess_image(processed_image):
    x = processed_image.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, axis=0)
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]  # Convert BGR to RGB
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

def style_loss(base_style, gram_target):
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target)) / (4.0 * (channels**2) * (width * height)**2)

def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    vectorized = tf.reshape(input_tensor, [-1, channels])
    gram = tf.matmul(tf.transpose(vectorized), vectorized)
    return gram


style_layers = [
    'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'
]
content_layers = ['block5_conv2']
all_layers = style_layers + content_layers

def get_model():
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in all_layers]
    model = Model([vgg.input], outputs)
    return model


def run_style_transfer(content_path, style_path, num_iterations=1000, style_weight=1e-2, content_weight=1e4):
    model = get_model()
    for layer in model.layers:
        layer.trainable = False

    content_image = load_and_process_image(content_path)
    style_image = load_and_process_image(style_path)
    generated_image = tf.Variable(content_image, dtype=tf.float32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)
    loss_weights = (style_weight, content_weight)

    # Extract features
    style_features = model(style_image)[:len(style_layers)]
    content_features = model(content_image)[len(style_layers):]

    gram_style_features = [gram_matrix(feature) for feature in style_features]

    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            all_loss = compute_loss(model, loss_weights, generated_image, gram_style_features, content_features)
        total_loss = all_loss[0]
        grads = tape.gradient(total_loss, generated_image)
        optimizer.apply_gradients([(grads, generated_image)])

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {total_loss.numpy()}")

    final_img = deprocess_image(generated_image.numpy())
    return final_img

if __name__ == "__main__":
    content_path = "images/content.jpg"
    style_path = "images/style.jpg"
    output = run_style_transfer(content_path, style_path)
    plt.imshow(output)
    plt.axis('off')
    plt.show()
