# Perceptual GAN for Small Object Detection

This project implements a Perceptual Generative Adversarial Network (GAN) to improve the detection of small objects in images. The core idea is to use a generator to enhance the features of small objects, making them more discriminative for a standard object detector.

## Project Structure

```
perceptual-gan-detection/
├── configs/
│   └── config.py           # Configuration file for all parameters
├── data/
│   ├── images/             # Directory for your image files
│   └── annotations.txt     # Annotation file for bounding boxes
├── src/
│   ├── data/
│   │   └── dataset.py      # PyTorch Dataset and DataLoader
│   ├── models/
│   │   ├── __init__.py
│   │   ├── detector.py     # Faster R-CNN detector model
│   │   ├── discriminator.py # Discriminator model for the GAN
│   │   └── generator.py    # Generator model for the GAN
│   ├── utils/
│   │   └── utils.py        # Utility functions (checkpoints, visualization)
│   └── train.py            # Main training script
└── README.md
```

## How It Works

The model consists of three main components:

1.  **Detector:** A Faster R-CNN model with a ResNet-50 backbone that serves as the primary object detector. It is also used to extract feature maps from the input images.
2.  **Generator:** A residual network that takes the feature maps of small objects (extracted by the detector's backbone) and attempts to generate super-resolved, more detailed features that resemble those of larger objects.
3.  **Discriminator:** A network that is trained to distinguish between the "real" features of large objects and the "fake" (generated) features of small objects.

The training process is a three-way adversarial game:
- The **Discriminator** learns to tell real and fake features apart.
- The **Generator** learns to fool the discriminator by making the generated features of small objects look like the real features of large objects.
- The **Detector** is trained on the original images and objects, but its training is influenced by the GAN, which helps it learn better representations for small objects.

## Files

### `configs/config.py`
This file contains all the hyperparameters and configuration settings for the project.

- **General Settings:** `DEVICE`, `SEED`.
- **Dataset & Data Loading:** `DATASET_PATH`, `ANNOTATION_FILE`, `NUM_CLASSES`, `BATCH_SIZE`, `NUM_WORKERS`.
- **Model & Training Parameters:** Learning rates for the Generator, Discriminator, and Detector, and the number of epochs.
- **Paths:** Directories for saving model checkpoints and visualization results.

### `src/data/dataset.py`
- **`PerceptualGAN_Dataset`**: A custom PyTorch `Dataset` class that loads images and their corresponding bounding box annotations from the `annotations.txt` file. It handles data validation, clipping bounding boxes to image boundaries, and mapping class labels.
- **`train_transform` / `test_transform`**: Defines the data augmentation pipelines for training and testing using the `albumentations` library. This includes resizing, flipping, and normalization.
- **`collate_fn`**: A function to correctly batch together images and targets, which can have varying numbers of objects.

### `src/models/`
- **`generator.py`**: Defines the `Generator` network, which consists of several `ResidualBlock`s to transform low-resolution features into high-resolution ones.
- **`discriminator.py`**: Defines the `Discriminator` network, a convolutional classifier that outputs a score indicating whether input features are real or fake.
- **`detector.py`**: Defines the `Detector` class, which wraps a pre-trained `fasterrcnn_resnet50_fpn` model from `torchvision` and customizes its final classification layer for the number of classes in our dataset.

### `src/utils/utils.py`
Contains helper functions for:
- `save_checkpoint` / `load_checkpoint`: Saving and loading model states.
- `draw_boxes`: Drawing bounding boxes on images for visualization.
- `visualize_results`: Creating a side-by-side comparison of ground truth and predicted bounding boxes and saving it as an image.

### `train.py`
This is the main script to run the training process.
- It initializes the dataset and data loaders.
- It creates instances of the Generator, Discriminator, and Detector models.
- It sets up the Adam optimizers for all three networks.
- The main training loop (`train_one_epoch`) alternates between training the discriminator, the generator, and the detector.
- After training, it saves the final models and runs inference on a test image to produce a visualization of the results.

## Installation

1.  **Clone the repository (or create the files as described above).**

2.  **Create a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    You will need PyTorch, Torchvision, Albumentations, and other common libraries.
    ```bash
    pip install torch torchvision torchaudio
    pip install numpy matplotlib pillow albumentations
    ```

## Usage

### 1. Prepare Your Data

- Place your images in the `data/images/` directory.
- Create an `annotations.txt` file in the `data/` directory. Each line should be in the format:
  `image_name.jpg,xmin,ymin,xmax,ymax,class_id`

  **Example:**
  ```
  0001.jpg,100,150,120,180,0
  0001.jpg,200,250,230,290,1
  0002.jpg,50,60,75,85,0
  ```

### 2. Configure Your Training

- Adjust the parameters in `configs/config.py` as needed (e.g., `NUM_CLASSES`, `BATCH_SIZE`, `NUM_EPOCHS`).

### 3. Run Training

- To start the training process, run the `train.py` script from the root of the `perceptual-gan-detection` directory:
  ```bash
  python train.py
  ```

### 4. View Results

- Model checkpoints will be saved in the `results/checkpoints/` directory.
- A visualization of the model's predictions on a test image will be saved as `results/visualizations/test_results.png`.
