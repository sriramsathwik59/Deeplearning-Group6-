import numpy as np
from tensorflow.keras.datasets import mnist

def get_mnist():
    (train_images, _), (test_images, _) = mnist.load_data()
    return np.vstack([train_images, test_images]).astype(np.float32) / 255.

def generate_moving_mnist(shape=(64, 64), seq_len=20, num_seqs=1000):
    mnist_data = get_mnist()
    width, height = shape
    dataset = np.zeros((num_seqs, seq_len, width, height), dtype=np.float32)

    for i in range(num_seqs):
        idxs = np.random.choice(mnist_data.shape[0], size=2, replace=False)
        digits = [mnist_data[idx] for idx in idxs]
        positions = [np.random.rand(2) * (np.array([width, height]) - 28) for _ in range(2)]
        velocities = [np.random.rand(2) * 0.2 - 0.1 for _ in range(2)]

        for t in range(seq_len):
            frame = np.zeros((width, height), dtype=np.float32)
            for digit, pos, vel in zip(digits, positions, velocities):
                pos += vel
                pos[0] = np.clip(pos[0], 0, width - 28)
                pos[1] = np.clip(pos[1], 0, height - 28)
                x, y = int(pos[0]), int(pos[1])
                frame[x:x+28, y:y+28] += digit  # Ensure this is adding correctly
            dataset[i, t] = np.clip(frame, 0, 1)  # Prevent overflow

    return dataset

# Generate the dataset
moving_mnist = generate_moving_mnist()
print("Generated Moving MNIST data shape:", moving_mnist.shape)

# Save the dataset to disk
np.save('moving_mnist.npy', moving_mnist)
print("Dataset saved to 'moving_mnist.npy'")
