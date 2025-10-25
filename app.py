from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing import image
from PIL import Image
import io, time, os, tempfile

app = Flask(__name__)

# ----------------------------
# Utility functions
# ----------------------------
def preprocess_image(img_file):
    img = Image.open(img_file).convert("RGB").resize((400, 400))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return vgg19.preprocess_input(img)

def deprocess_image(x):
    x = x.reshape((400, 400, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def gram_matrix(tensor):
    x = tf.transpose(tensor, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

# ----------------------------
# Model setup
# ----------------------------
content_layers = ['block5_conv2']
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

vgg = vgg19.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False
outputs = [vgg.get_layer(name).output for name in style_layers + content_layers]
model = tf.keras.Model([vgg.input], outputs)

# ----------------------------
# Loss function
# ----------------------------
def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights
    model_outputs = model(init_image)
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        gram_comb_style = gram_matrix(comb_style[0])
        style_score += weight_per_style_layer * tf.reduce_mean(tf.square(gram_comb_style - target_style))

    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += tf.reduce_mean(tf.square(comb_content[0] - target_content))

    loss = style_weight * style_score + content_weight * content_score
    return loss

# ----------------------------
# Style Transfer Route
# ----------------------------
@app.route('/')
def home():
    return "Neural Style Transfer API is running! Use POST /transfer with 'content' and 'style' images."

@app.route('/transfer', methods=['POST'])
def transfer_style():
    if 'content' not in request.files or 'style' not in request.files:
        return jsonify({"error": "Please upload both 'content' and 'style' images"}), 400

    content_file = request.files['content']
    style_file = request.files['style']

    content_image = preprocess_image(content_file)
    style_image = preprocess_image(style_file)

    # Extract features
    style_outputs = model(style_image)
    content_outputs = model(content_image)
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    # Initialize generated image
    generated_image = tf.Variable(content_image, dtype=tf.float32)
    opt = tf.optimizers.Adam(learning_rate=5.0)
    loss_weights = (1e-2, 1e4)

    best_loss, best_img = float('inf'), None
    epochs = 150

    for i in range(epochs):
        with tf.GradientTape() as tape:
            loss = compute_loss(model, loss_weights, generated_image, gram_style_features, content_features)
        grad = tape.gradient(loss, generated_image)
        opt.apply_gradients([(grad, generated_image)])
        if loss < best_loss:
            best_loss = loss
            best_img = generated_image.numpy()
        if i % 50 == 0:
            print(f"Iteration {i}, Loss: {loss:.2f}")

    # Convert to image and send back
    final_img = deprocess_image(best_img)
    result = Image.fromarray(final_img)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    result.save(temp_file.name)
    temp_file.close()

    return send_file(temp_file.name, mimetype='image/jpeg')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
