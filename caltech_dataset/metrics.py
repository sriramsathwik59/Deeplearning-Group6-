import os
import cv2
import numpy as np
import imageio
from pytorch_msssim import ssim
import torch
import imageio

def load_gif_frames(gif_path):
    frames = []
    with imageio.get_reader(gif_path) as reader:
        for frame in reader:
            # Convert the frame to an array and append to the list
            frames.append(frame)
    return frames

def load_video_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def calculate_metrics(predicted_frames, ground_truth_frames):
    predicted_frames = np.array(predicted_frames)
    ground_truth_frames = np.array(ground_truth_frames)
    
    print("Shape of predicted_frames:", predicted_frames.shape)
    print("Shape of ground_truth_frames:", ground_truth_frames.shape)
    
    # Ensure the same number of frames
    min_frames = min(predicted_frames.shape[0], ground_truth_frames.shape[0])
    predicted_frames = predicted_frames[:min_frames]
    ground_truth_frames = ground_truth_frames[:min_frames]
    
    mse = np.mean((predicted_frames - ground_truth_frames) ** 2)
    psnr = 10 * np.log10(1.0 / mse)
    predicted_frames_tensor = torch.from_numpy(predicted_frames).permute(0, 3, 1, 2).float()
    ground_truth_frames_tensor = torch.from_numpy(ground_truth_frames).permute(0, 3, 1, 2).float()
    ssim_scores = ssim(predicted_frames_tensor, ground_truth_frames_tensor, data_range=255, size_average=False)
    ssim_avg = ssim_scores.mean().item()
    return mse, psnr, ssim_avg

# Set the paths to the predicted GIF and original video
predicted_gif_path = "/home/ubuntu/project/V010_predicted.gif"
original_video_path = "/home/ubuntu/project/V010_original.mp4"

# Load the predicted frames from the GIF and the ground truth frames from the video
predicted_frames = load_gif_frames(predicted_gif_path)
ground_truth_frames = load_video_frames(original_video_path)

# Calculate the metrics
mse, psnr, ssim_avg = calculate_metrics(predicted_frames, ground_truth_frames)

# Print the results
print(f"MSE: {mse:.4f}, PSNR: {psnr:.2f}, SSIM: {ssim_avg:.4f}")