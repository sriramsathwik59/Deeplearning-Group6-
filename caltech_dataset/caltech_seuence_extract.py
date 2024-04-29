import os
import cv2

# Set the path to the train and test folders
train_root = "/home/ubuntu/project/dataset/Train/"
test_root = "/home/ubuntu/project/dataset/Test/"

# Set the names of the train and test sets
train_sets = ["set00", "set01", "set02", "set03", "set04", "set05"]
test_sets = ["set06", "set07", "set08", "set09", "set10"]

def convert_seq_to_images(set_path):
    # Get the list of .seq files in the set directory
    seq_files = [f for f in os.listdir(set_path) if f.endswith(".seq")]
    
    # Loop over each .seq file in the set
    for seq_file in seq_files:
        # Extract the sequence name from the file name
        seq_name = os.path.splitext(seq_file)[0]
        
        # Set the destination path for the images
        save_dest = os.path.join(set_path, seq_name)
        
        # Create the destination directory if it doesn't exist
        os.makedirs(save_dest, exist_ok=True)
        
        # Read .seq file using OpenCV
        seq = cv2.VideoCapture(os.path.join(set_path, seq_file))
        
        frame_count = 0
        
        # Read and save frames
        while True:
            ret, frame = seq.read()
            if not ret:
                break
            
            # Save the frame as an image file
            img_file = f"image{frame_count+1:05d}.jpg"
            img_path = os.path.join(save_dest, img_file)
            cv2.imwrite(img_path, frame)
            
            frame_count += 1
        
        seq.release()

# Convert .seq files to images for the train sets
for train_set in train_sets:
    set_path = os.path.join(train_root, train_set, train_set)
    convert_seq_to_images(set_path)

# Convert .seq files to images for the test sets
for test_set in test_sets:
    set_path = os.path.join(test_root, test_set, test_set)
    convert_seq_to_images(set_path)