# Project Setup Instructions

This README provides step-by-step instructions to set up and run the project.

## Step 1: Clone the Repository
First, clone the repository to your local machine.

```bash
git clone <repository-url>
```

## Step 2: Navigate to the Repository Directory
Change directory to the root of the cloned repository.

```bash
cd Deeplearning-Group6-
```

## Step 3: Navigate to the Final_Code Directory
Change directory to `Final_Code` which contains the project code.

```bash
cd Final_Code
```

## Step 4: Download the Dataset
Download the dataset directly into the `Final_Code` folder using the following command:

```bash
wget https://data.caltech.edu/records/f6rph-90m20/files/data_and_labels.zip?download=1 -O data_and_labels.zip
```

## Step 5: Unzip the Dataset
Unzip the dataset.

```bash
unzip data_and_labels.zip?download=1.zip
```

## Step 6: Run sequence.py
Run `sequence.py`. Make sure you give the correct path of the data folder and output accordingly.

```bash
cd ..
python3 sequence.py
```

## Step 7: Run train.py
Run `train.py` to begin training your model. Ensure you give the correct path of the train folder accordingly.

```bash
python3 train.py
```

## Step 8: Run test.py
Run `test.py` to execute the testing phase. Make sure the paths to the test data are correctly specified.

```bash
python3 test.py
```

Follow these steps to ensure your project is set up correctly and ready to run.
