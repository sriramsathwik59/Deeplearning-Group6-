import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set the image size and number of channels
img_size = (320, 320)
num_channels = 3

# Set the number of frames to use for both input and prediction
num_frames = 6

# Set the batch size and number of epochs
batch_size = 1
num_epochs = 1

# Path to the folder containing images
image_folder = '/home/ubuntu/Train_Images/'

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

def load_image(image_path):
    # Read the image file
    image = cv2.imread(image_path)

    # Convert the image from BGR to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to the desired size
    image = cv2.resize(image, img_size)

    # Normalize the pixel values to the range [0, 1]
    image = image.astype(np.float32) / 255.0

    return image

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        # Define the CNN architecture
        self.layers = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.layers(x)

class VideoPredictionModel(nn.Module):
    def __init__(self, num_frames):
        super(VideoPredictionModel, self).__init__()
        self.feature_extractor = CustomCNN()
        self.lstm = nn.LSTM(input_size=256, hidden_size=512, num_layers=2, batch_first=True)

        # Fix: Correct the number of output elements for the output layer
        # 320*320 pixels * 3 color channels per output frame
        output_size = num_frames * img_size[0] * img_size[1] * num_channels
        self.output_layer = nn.Linear(512, output_size)  # Update this line accordingly

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(-1, channels, height, width)
        features = self.feature_extractor(x)
        features = features.view(batch_size, num_frames, -1)
        lstm_output, _ = self.lstm(features)

        # Forward through the output layer
        output = self.output_layer(lstm_output[:, -1])  # Use the last time step's output

        # Reshape to match the desired output shape (batch, num_frames, channels, height, width)
        output = output.view(batch_size, num_frames, channels, height, width)
        return output

class VideoDataset(Dataset):
    def __init__(self, folder, num_frames):
        self.folder = folder
        self.num_frames = num_frames
        self.image_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")])
        self.num_images = len(self.image_files)

    def __len__(self):
        return max(0, self.num_images - 2 * self.num_frames + 1)

    def __getitem__(self, index):
        frame_paths = self.image_files[index:index + 2 * self.num_frames]
        input_frames = [load_image(path) for path in frame_paths[:self.num_frames]]
        output_frames = [load_image(path) for path in frame_paths[self.num_frames:]]
        input_frames = torch.from_numpy(np.array(input_frames)).permute(0, 3, 1, 2).float().to(device)
        output_frames = torch.from_numpy(np.array(output_frames)).permute(0, 3, 1, 2).float().to(device)
        return input_frames, output_frames

# Create the dataset and model
dataset = VideoDataset(image_folder, num_frames)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = VideoPredictionModel(num_frames).to(device)

# Training setup
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
for epoch in range(num_epochs):
    for batch_idx, (input_frames, output_frames) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_frames = model(input_frames)
        loss = criterion(predicted_frames, output_frames)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), "CustomCNN_video_prediction.pth")
print("Model saved as 'CustomCNN_video_prediction.pth'")