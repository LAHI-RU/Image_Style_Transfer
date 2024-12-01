import tensorflow as tf
from tensorflow.keras.applications import vgg19
from tensorflow.keras.models import Model
import numpy as np
import PIL.Image
import os

# --- Helper Functions ---
def load_and_process_image(image_path, target_size=(512, 512)):
    """Load and preprocess an image for the VGG19 model."""
    img = tf.keras.utils.load_img(image_path, target_size=target_size)
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = vgg19.preprocess_input(img)  # VGG19 preprocessing
    return tf.convert_to_tensor(img)

def deprocess_image(processed_img):
    """De-process an image after style transfer."""
    x = processed_img.copy()
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]  # Convert BGR to RGB
    x = np.clip(x, 0, 255).astype("uint8")
    return x

def compute_gram_matrix(tensor):
    """Compute the Gram matrix for a given tensor."""
    channels = int(tensor.shape[-1])
    vectorized = tf.reshape(tensor, [-1, channels])
    gram_matrix = tf.matmul(vectorized, vectorized, transpose_a=True)
    return gram_matrix / tf.cast(tf.shape(vectorized)[0], tf.float32)

# --- Loss Functions ---
def compute_loss(model, loss_weights, generated_image, gram_style_features, content_features):
    """Compute total loss for style transfer."""
    content_weight, style_weight = loss_weights
    model_outputs = model(generated_image)
    generated_content_features = model_outputs[:len(content_features)]
    generated_style_features = model_outputs[len(content_features):]
    
    # Content loss
    content_loss = tf.add_n([
        tf.reduce_mean(tf.square(content_feature - gen_content_feature))
        for content_feature, gen_content_feature in zip(content_features, generated_content_features)
    ])
    content_loss *= content_weight

    # Style loss
    style_loss = tf.add_n([
        tf.reduce_mean(tf.square(gram_style_feature - compute_gram_matrix(gen_style_feature)))
        for gram_style_feature, gen_style_feature in zip(gram_style_features, generated_style_features)
    ])
    style_loss *= style_weight

    # Total variation loss (smoothness)
    total_variation_loss = tf.image.total_variation(generated_image)

    # Combine losses
    total_loss = content_loss + style_loss + total_variation_loss
    return total_loss, (style_loss, content_loss, total_variation_loss)

# --- Load VGG19 Model ---
def get_model():
    """Load the VGG19 model and return intermediate layers for style and content."""
    vgg = vgg19.VGG19(weights="imagenet", include_top=False)
    vgg.trainable = False

    # Content layer
    content_layers = ["block5_conv2"]
    # Style layers
    style_layers = [
        "block1_conv1", "block2_conv1",
        "block3_conv1", "block4_conv1", "block5_conv1"
    ]
    selected_layers = style_layers + content_layers
    outputs = [vgg.get_layer(name).output for name in selected_layers]
    return Model([vgg.input], outputs), len(style_layers)

# --- Main Function for Style Transfer ---
def run_style_transfer(content_path, style_path, iterations=1000, content_weight=1e4, style_weight=1e-2):
    """Run the style transfer process."""
    # Load and preprocess images
    content_image = load_and_process_image(content_path)
    style_image = load_and_process_image(style_path)

    # Load the VGG19 model
    model, num_style_layers = get_model()

    # Extract features
    content_features = model(content_image)[-1:]  # Only content layers
    style_features = model(style_image)[:num_style_layers]  # Only style layers
    gram_style_features = [compute_gram_matrix(feature) for feature in style_features]

    # Initialize the generated image
    generated_image = tf.Variable(content_image, trainable=True)

    # Define optimizer
    optimizer = tf.optimizers.Adam(learning_rate=5.0)

    # Loss weights
    loss_weights = (content_weight, style_weight)

    # Optimization loop
    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            total_loss, _ = compute_loss(
                model, loss_weights, generated_image,
                gram_style_features, content_features
            )
        gradients = tape.gradient(total_loss, generated_image)
        optimizer.apply_gradients([(gradients, generated_image)])
        return total_loss

    for i in range(iterations):
        total_loss = train_step()

        # Print progress every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}/{iterations}, Loss: {total_loss.numpy()}")

    # Deprocess the final image
    final_img = deprocess_image(generated_image.numpy()[0])
    return final_img

# --- Save Output ---
def save_image(output_image, output_path):
    """Save the generated image to a file."""
    output_image = PIL.Image.fromarray(output_image)
    output_image.save(output_path)

# --- Main Execution ---
if __name__ == "__main__":
    # Paths to content and style images
    content_path = "images/content.jpg"
    style_path = "images/style.jpg"
    output_path = "images/output.jpg"

    # Run style transfer
    final_image = run_style_transfer(content_path, style_path, iterations=1000)
    save_image(final_image, output_path)
    print("Style transfer complete. Output saved at:", output_path)
