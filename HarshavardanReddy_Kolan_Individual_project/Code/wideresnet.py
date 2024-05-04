import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import wide_resnet38_2

# Set the path to the train folder
train_folder = "/home/ubuntu/project/dataset/Train/"

# Set the image size and number of channels
img_size = (320, 320)
num_channels = 3

# Set the number of frames to use for input and prediction
num_input_frames = 4
num_output_frames = 1

# Set the batch size and number of epochs
batch_size = 30
num_epochs = 1

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")


class VideoPredictionModel(nn.Module):
    def __init__(self, num_input_frames, num_output_frames):
        super(VideoPredictionModel, self).__init__()

        # Load pre-trained WideResNet38 model without the final classification layer
        self.feature_extractor = wide_resnet38_2(pretrained=True, progress=True)
        self.feature_extractor.fc = nn.Identity()  # Remove final classification layer

        # Sequence modeling component (LSTM)
        self.lstm = nn.LSTM(input_size=2048, hidden_size=512, num_layers=2, batch_first=True)

        # Output layer
        self.output_layer = nn.Linear(512, num_output_frames * num_channels * np.prod(img_size))

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()

        # Reshape input to (batch_size * num_frames, channels, height, width)
        x = x.view(-1, channels, height, width)

        # Extract features using WideResNet38
        features = self.feature_extractor(x)

        # Reshape features to (batch_size, num_frames, feature_size)
        features = features.view(batch_size, num_frames, -1)

        # Apply LSTM to capture temporal dependencies
        lstm_output, _ = self.lstm(features)

        # Apply output layer
        output = self.output_layer(lstm_output)

        # Reshape output to (batch_size, num_output_frames, channels, height, width)
        output = output.view(batch_size, num_output_frames, channels, height, width)

        return output


class VideoDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.train_data = []

        # Specify the train sets to use
        train_sets = ["set00", "set01"]

        # Iterate over the specified train sets
        for train_set in train_sets:
            print(f"Processing train set: {train_set}")
            set_folder = os.path.join(self.folder, train_set, train_set)

            # Get the list of sequence folders
            sequence_folders = [os.path.join(set_folder, f) for f in os.listdir(set_folder) if f.startswith("V")]
            print(f"Found {len(sequence_folders)} sequence folders in {train_set}")

            # Iterate over the sequence folders
            for sequence_folder in sequence_folders:
                print(f"Processing sequence folder: {sequence_folder}")

                # Get the list of image files in the sequence folder
                image_files = sorted([f for f in os.listdir(sequence_folder) if f.endswith(".jpg")])
                print(f"Found {len(image_files)} image files in {sequence_folder}")

                # Create input-output pairs for training
                for i in range(len(image_files) - num_input_frames - num_output_frames + 1):
                    input_frames = []
                    output_frames = []

                    # Store the paths of input frames
                    for j in range(i, i + num_input_frames):
                        image_path = os.path.join(sequence_folder, image_files[j])
                        input_frames.append(image_path)

                    # Store the paths of output frames
                    for j in range(i + num_input_frames, i + num_input_frames + num_output_frames):
                        image_path = os.path.join(sequence_folder, image_files[j])
                        output_frames.append(image_path)

                    # Add the input-output pair to the training data
                    self.train_data.append((input_frames, output_frames))

        print("Dataset creation completed.")

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        input_frames_paths, output_frames_paths = self.train_data[index]

        # Load input frames
        input_frames = []
        for path in input_frames_paths:
            frame = load_image(path)
            input_frames.append(frame)

        # Load output frames
        output_frames = []
        for path in output_frames_paths:
            frame = load_image(path)
            output_frames.append(frame)

        input_frames = torch.from_numpy(np.array(input_frames)).permute(0, 3, 1, 2).float().to(device)
        output_frames = torch.from_numpy(np.array(output_frames)).permute(0, 3, 1, 2).float().to(device)

        return input_frames, output_frames


# Create the dataset
print("Creating the dataset...")
dataset = VideoDataset(train_folder)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create the VideoPredictionModel
model = VideoPredictionModel(num_input_frames, num_output_frames).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
print("Starting training...")
for epoch in range(num_epochs):
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    for batch_idx, (input_frames, output_frames) in enumerate(dataloader):
        input_frames = input_frames.to(device)
        output_frames = output_frames.to(device)

        optimizer.zero_grad()

        # Forward pass
        predicted_frames = model(input_frames)

        # Compute the loss
        loss = criterion(predicted_frames, output_frames)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            print(f"Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Loss: {loss.item():.4f}")

# Save the trained model
print("Training completed. Saving the model...")
torch.save(model.state_dict(), "WideResNet38_video_prediction.pth")
print("Model saved as 'WideResNet38_video_prediction.pth'")