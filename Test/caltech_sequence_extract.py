import os
import cv2
current_directory = os.getcwd()
# Set the path to the train and test folders
train_root = os.path.join(current_directory, "dataset/Train")
test_root = os.path.join(current_directory, "dataset/Test")

# Set the names of the train and test sets
train_sets = ["set00"]
test_sets = ["set06"]

# Destination directories for saving train and test images
train_output_dir = "/home/ubuntu/Train_Images/"
test_output_dir = "/home/ubuntu/Test_Images/"

# Create the destination directories if they don't exist
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)


def convert_seq_to_images(set_path, output_dir):
    # Get the list of .seq files in the set directory
    seq_files = [f for f in os.listdir(set_path) if f.endswith(".seq")]

    # Loop over each .seq file in the set
    for seq_file in seq_files:
        # Extract the sequence name from the file name
        seq_name = os.path.splitext(seq_file)[0]

        # Read .seq file using OpenCV
        seq = cv2.VideoCapture(os.path.join(set_path, seq_file))

        frame_count = 0

        # Read and save frames
        while True:
            ret, frame = seq.read()
            if not ret:
                break

            # Save the frame as an image file
            img_file = f"{seq_name}_image{frame_count + 1:05d}.jpg"
            img_path = os.path.join(output_dir, img_file)
            cv2.imwrite(img_path, frame)

            frame_count += 1

        seq.release()


# Convert .seq files to images for the train sets
for train_set in train_sets:
    set_path = os.path.join(train_root, train_set, train_set)
    convert_seq_to_images(set_path, train_output_dir)

# Convert .seq files to images for the test sets
for test_set in test_sets:
    set_path = os.path.join(test_root, test_set, test_set)
    convert_seq_to_images(set_path, test_output_dir)
