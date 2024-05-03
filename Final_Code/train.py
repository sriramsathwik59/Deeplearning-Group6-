import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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
class PredNet(nn.Module):
    def __init__(self, input_shape):
        super(PredNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0] * input_shape[3], 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.reshape(batch_size, num_frames * channels, height, width)

        x = self.encoder(x)
        x = self.decoder(x)

        x = x.reshape(batch_size, num_output_frames, channels, height, width)
        return x

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

# Create the PredNet model
input_shape = (num_input_frames, img_size[0], img_size[1], num_channels)
model = PredNet(input_shape).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
print("Starting training...")
for epoch in range(num_epochs):
    print(f"Epoch [{epoch+1}/{num_epochs}]")
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
            print(f"Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
    
    print(f"Epoch [{epoch+1}/{num_epochs}] completed. Loss: {loss.item():.4f}")

# Save the trained model
print("Training completed. Saving the model...")
torch.save(model.state_dict(), "Model_320")
print("Model saved as 'prednet_model_set00_set01.pth'")