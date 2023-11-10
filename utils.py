from PIL import Image
import numpy as np
import os
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
import shutil

def load_checkpoint(filepath):
    print(torch.__version__)
    print(torch.cuda.is_available()) 
    checkpoint = torch.load(filepath)
    
    # Load the model architecture
    model = models.vgg16(pretrained=True)
    classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 102),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier
    
    # Load the model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load the class-to-index mapping
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model, checkpoint['hyperparameters']


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    # Open and resize the image while maintaining the aspect ratio
    im = Image.open(image)
    im = im.resize((256, 256))
    
    # Crop the center 224x224 portion of the image
    left = (256 - 224) / 2
    top = (256 - 224) / 2
    right = (256 + 224) / 2
    bottom = (256 + 224) / 2
    im = im.crop((left, top, right, bottom))
    
    # Convert color channels from 0-255 to 0-1
    np_image = np.array(im) / 255.0
    
    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Reorder dimensions and transpose the color channel
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
     # Set the model to evaluation mode
    model.eval()
    
    # Preprocess the image
    image = process_image(image_path)
    
    # Convert the NumPy array to a PyTorch tensor
    image = torch.from_numpy(image).float()
    
    # Add a batch dimension
    image = image.unsqueeze(0)
    
    # Use a model to make predictions
    with torch.no_grad():
        output = model(image)
    
    # Calculate class probabilities and indices of the topk classes
    probs, indices = torch.topk(torch.exp(output), topk)
    
    # Convert probabilities and indices to lists
    probs = probs[0].numpy()
    indices = indices[0].numpy()
    
    # Map indices to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[i] for i in indices]
    
    return probs, classes

# TODO: Display an image along with the top 5 classes
def display_prediction(image_path, model, cat_to_name, topk):
    # Make predictions
    probs, classes = predict(image_path, model, topk)
    
    # Map class labels to flower names using cat_to_name
    class_names = [cat_to_name[c] for c in classes]
    
    # Display the image and predictions
    image = Image.open(image_path)
    
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(image)
    ax1.axis('off')
    ax2.barh(np.arange(len(class_names)), probs)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(len(class_names)))
    ax2.set_yticklabels(class_names)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.show()  


