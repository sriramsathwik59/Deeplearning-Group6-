import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import imageio

# Set the path to the test folder
test_folder = "/home/ubuntu/project/dataset/Test/"

# Set the image size and number of channels
img_size = (320, 320)
num_channels = 3

# Set the number of frames to use for input and prediction
num_input_frames = 4
num_output_frames = 1

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# Specify the test sets to use
test_sets = ["set06", "set07"]

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

class TestDataset(Dataset):
    def __init__(self, folder, test_sets):
        self.folder = folder
        self.test_data = []
        
        # Iterate over the specified test sets
        for test_set in test_sets:
            set_folder = os.path.join(self.folder, test_set, test_set)
            
            # Get the list of sequence folders
            sequence_folders = [os.path.join(set_folder, f) for f in os.listdir(set_folder) if f.startswith("V")]
            
            # Iterate over the sequence folders
            for sequence_folder in sequence_folders:
                # Get the list of image files in the sequence folder
                image_files = sorted([f for f in os.listdir(sequence_folder) if f.endswith(".jpg")])
                
                # Create input-output pairs for testing
                for i in range(len(image_files) - num_input_frames):
                    input_frames = []
                    
                    # Store the paths of input frames
                    for j in range(i, i + num_input_frames):
                        image_path = os.path.join(sequence_folder, image_files[j])
                        input_frames.append(image_path)
                    
                    # Add the input frames and the sequence folder path
                    self.test_data.append((input_frames, sequence_folder))
    
    def __len__(self):
        return len(self.test_data)
    
    def __getitem__(self, index):
        input_frames_paths, sequence_folder = self.test_data[index]
        
        # Load input frames
        input_frames = []
        for path in input_frames_paths:
            frame = load_image(path)
            input_frames.append(frame)
        
        input_frames = torch.from_numpy(np.array(input_frames)).permute(0, 3, 1, 2).float().to(device)
        
        return input_frames, sequence_folder

def predict_frames(model, sequence_folder):
    predicted_frames = []
    image_files = sorted([f for f in os.listdir(sequence_folder) if f.endswith(".jpg")])
    
    for i in range(0, len(image_files) - num_input_frames, num_input_frames):
        input_frames = []
        
        # Load the input frames
        for j in range(i, i + num_input_frames):
            image_path = os.path.join(sequence_folder, image_files[j])
            frame = load_image(image_path)
            input_frames.append(frame)
        
        input_frames = torch.from_numpy(np.array(input_frames)).permute(0, 3, 1, 2).float().to(device)
        
        with torch.no_grad():
            output = model(input_frames.unsqueeze(0))
            output = output.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
            output = (output * 255).astype(np.uint8)
            predicted_frames.append(output[0])
    
    return predicted_frames

def create_gif(frames, output_path):
    imageio.mimsave(output_path, frames, duration=0.1)

# Load the trained model
input_shape = (num_input_frames, img_size[0], img_size[1], num_channels)
model = PredNet(input_shape).to(device)
model.load_state_dict(torch.load("Model_320"))
model.eval()

# Create the test dataset
test_dataset = TestDataset(test_folder, test_sets)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Predict frames for each test sequence
output_dir = "/home/ubuntu/project/"
processed_sequences = set()

for data in test_dataloader:
    input_frames, sequence_folder = data
    sequence_folder = sequence_folder[0]  # Extract the sequence folder path from the tuple
    
    if sequence_folder not in processed_sequences:
        # Predict the frames
        separated_predicted_frames = predict_frames(model, sequence_folder)
        
        # Create a GIF or video of the predicted frames
        output_path = os.path.join(output_dir, os.path.basename(sequence_folder) + ".gif")
        create_gif(separated_predicted_frames, output_path)
        print(f"GIF/video created for sequence: {sequence_folder}")
        
        processed_sequences.add(sequence_folder)  # Mark the sequence folder as processed

print("Prediction completed.") 