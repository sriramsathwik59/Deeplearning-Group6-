import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from PIL import Image

# Load images from a directory and preprocess them
def load_images(image_dir, image_size=(64, 64)):
    images = []
    for img_path in glob.glob(os.path.join(image_dir, '*.jpg')):  # Adjust the pattern to match your images' format
        img = Image.open(img_path)
        img = img.resize(image_size, Image.Resampling.LANCZOS)  # Updated line
        img = np.array(img)
        if img.shape == (image_size[0], image_size[1], 3):
            images.append(img)
    images = np.array(images)
    images = (images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    return images

# Define the generator model
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(8*8*256, use_bias=False, input_shape=(100,)),  # Adjust input dimension if needed
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((8, 8, 256)),  # Adjust this to match the expected output shape
        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Input(shape=(32, 32, 3)),  # Ensure this matches generator output size
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Compile the GAN
def compile_gan(generator, discriminator):
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    discriminator.trainable = False
    gan_input = layers.Input(shape=(100,))
    generated_frame = generator(gan_input)
    gan_output = discriminator(generated_frame)
    gan = Model(gan_input, gan_output)
    gan.compile(optimizer='adam', loss='binary_crossentropy')
    return gan

# Train the GAN
def train(generator, discriminator, gan, real_frames, epochs=50, batch_size=32, noise_dim=100):
    for epoch in range(epochs):
        for _ in range(real_frames.shape[0] // batch_size):
            noise = np.random.normal(0, 1, (batch_size, noise_dim))
            fake_data = generator.predict(noise)

            idx = np.random.randint(0, real_frames.shape[0], batch_size)
            real_data = real_frames[idx]

            real_y = np.ones((batch_size, 1))
            fake_y = np.zeros((batch_size, 1))

            d_loss_real = discriminator.train_on_batch(real_data, real_y)
            d_loss_fake = discriminator.train_on_batch(fake_data, fake_y)

            noise = np.random.normal(0, 1, (batch_size, noise_dim))
            g_loss = gan.train_on_batch(noise, real_y)

        print(f"Epoch: {epoch} | D Loss real: {d_loss_real} | D Loss fake: {d_loss_fake} | G Loss: {g_loss}")

# Main function to run the GAN
def main():
    image_dir = '/home/ubuntu/Train/set00/set00/V000/'  # Specify the path to your images
    images = load_images(image_dir)
    generator = build_generator()
    discriminator = build_discriminator()
    gan = compile_gan(generator, discriminator)
    train(generator, discriminator, gan, images)

if __name__ == "__main__":
    main()
