import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import torch.nn as nn
import torch.optim as optim

# Parameters
img_size = (320, 320)
num_channels = 3
num_frames = 6  # Number of frames used for both input and output
batch_size = 1

# Path to the folder containing test images
test_image_folder = '/home/ubuntu/Test/set06/set06/V000/images/'

# Output directory for saved images and videos
output_dir = '/home/ubuntu/output/'
os.makedirs(output_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing on device: {device}")

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, img_size)
    image = image.astype(np.float32) / 255.0
    return image

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
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
        output_size = num_frames * img_size[0] * img_size[1] * num_channels
        self.output_layer = nn.Linear(512, output_size)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(-1, channels, height, width)
        features = self.feature_extractor(x)
        features = features.view(batch_size, num_frames, -1)
        lstm_output, _ = self.lstm(features)
        output = self.output_layer(lstm_output[:, -1])
        output = output.view(batch_size, num_frames, channels, height, width)
        return output

class TestDataset(Dataset):
    def __init__(self, folder, num_frames):
        self.folder = folder
        self.num_frames = num_frames
        self.image_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")])
        self.num_images = len(self.image_files)

    def __len__(self):
        return max(0, self.num_images - 2 * self.num_frames + 1)

    def __getitem__(self, index):
        input_paths = self.image_files[index:index + self.num_frames]
        input_frames = [load_image(path) for path in input_paths]
        input_frames = torch.from_numpy(np.array(input_frames)).permute(0, 3, 1, 2).float().to(device)

        target_paths = self.image_files[index + self.num_frames:index + 2 * self.num_frames]
        target_frames = [load_image(path) for path in target_paths]
        target_frames = torch.from_numpy(np.array(target_frames)).permute(0, 3, 1, 2).float().to(device)

        return input_frames, target_frames

# Load the trained model
model_path = '/home/ubuntu/Final_Code/CustomCNN_video_prediction.pth'
model = VideoPredictionModel(num_frames).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Create the test dataset and dataloader
test_dataset = TestDataset(test_image_folder, num_frames)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def compute_metrics(predicted, target):
    mse = F.mse_loss(predicted, target).item()
    psnr = compare_psnr(target.cpu().numpy(), predicted.cpu().numpy(), data_range=1)
    return mse, psnr

def save_frames(frames, folder, prefix):
    os.makedirs(folder, exist_ok=True)
    for i, frame in enumerate(frames):
        image = frame.cpu().permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(folder, f"{prefix}_frame_{i:04d}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

print("Starting testing...")
predicted_video_frames_folder = os.path.join(output_dir, "predicted_frames")
with torch.no_grad():
    total_mse = 0
    total_psnr = 0
    for batch_idx, (input_frames, target_frames) in enumerate(test_dataloader):
        predicted_frames = model(input_frames)
        mse, psnr = compute_metrics(predicted_frames, target_frames)
        total_mse += mse
        total_psnr += psnr
        print(f"Batch [{batch_idx + 1}/{len(test_dataloader)}] - MSE: {mse:.4f}, PSNR: {psnr:.2f}")

        # Save predicted frames for video generation
        save_frames(predicted_frames[0], predicted_video_frames_folder, f"batch_{batch_idx+1}")

print("Testing completed.")
print(f"Average MSE: {total_mse / len(test_dataloader):.4f}, Average PSNR: {total_psnr / len(test_dataloader):.2f}")

# Generate video from saved frames
def generate_video_from_frames(frame_folder, output_video_path, fps=30):
    """
    Generate a video from image frames stored in a folder.
    Args:
    frame_folder (str): Folder containing the image frames.
    output_video_path (str): Path to save the generated video.
    fps (int): Frames per second of the output video.
    """
    images = [img for img in sorted(os.listdir(frame_folder)) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(frame_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for image in images:
        img = cv2.imread(os.path.join(frame_folder, image))
        video.write(img)
    video.release()
    print(f"Video saved as {output_video_path}")

predicted_video_path = os.path.join(output_dir, "predicted_video.mp4")
generate_video_from_frames(predicted_video_frames_folder, predicted_video_path, fps=30)
