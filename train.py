import argparse
from torchvision import datasets, transforms, models
import torch
from torch import nn, optim
from collections import OrderedDict
import os
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch.nn.functional as F
import shutil
# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_directory', type=str, help='Path to the data directory')
parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
parser.add_argument('--arch', type=str, default='vgg16', help='Architecture (vgg13 or vgg16)')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
args = parser.parse_args()

# Check if GPU is available
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

model_names = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "vgg16": models.vgg16,
    "vgg13": models.vgg13
}

# Data transformation and loading

train_dir = os.path.join(args.data_directory, 'train')
valid_dir = os.path.join(args.data_directory, 'valid')
test_dir = os.path.join(args.data_directory, 'test')
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Load the datasets with ImageFolder. Here the images are loaded into memory with all the above transformations applied. 
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
}

# Using the image datasets and the transforms, define the dataloaders
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
    'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
}
print(len(dataloaders))


# Create model
if args.arch in model_names:
    # Load the selected model
    model = model_names[args.arch](pretrained=True)
else:
    print("Invalid model name. Please choose from the available models.")

# Modify the classifier with custom architecture
# Freeze the pre-trained network parameters so we don't backpropagate through them
for param in model.parameters():
    param.requires_grad = False

# Define a new, untrained feed-forward network as a classifier

classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 102),  #102 classes in the dataset
    nn.LogSoftmax(dim=1)
)

model.classifier = classifier

# Define loss and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr= args.learning_rate)

# Define number of epochs and device
num_epochs = args.epochs

# Move the model to the appropriate device
model.to(device)

# Define the number of steps for printing and the running loss
print_every = 40
running_loss = 0

for epoch in range(num_epochs):
    print(epoch)
    model.train()  # Set the model to training mode
    for inputs, labels in dataloaders['train']:
        print(labels)
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if running_loss % print_every == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/print_every:.4f}")
            running_loss = 0

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    accuracy = 0
    valid_loss = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['valid']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            valid_loss += criterion(outputs, labels).item()
            
            # Calculate accuracy
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = (top_class == labels.view(*top_class.shape))
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    # Calculate validation loss and accuracy
    avg_valid_loss = valid_loss / len(dataloaders['valid'])
    avg_accuracy = accuracy / len(dataloaders['valid'])

    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_valid_loss:.4f}, Validation Accuracy: {avg_accuracy:.4f}")

# Save hyperparameters (learning rate, units in the classifier, epochs, etc) for the next part
hyperparameters = {
    'learning_rate': args.learning_rate,
    'hidden_units': args.hidden_units,
    'epochs': num_epochs
}


# Save the trained model as a checkpoint

# Define the path for saving the checkpoint file
checkpoint_path = args.save_dir + '/checkpoint.pth'

# Save the model state, class-to-index mapping, and optimizer state
checkpoint = {
    'model_state_dict': model.state_dict(),
    'class_to_idx': image_datasets['train'].class_to_idx,
    'optimizer_state_dict': optimizer.state_dict(),
    'hyperparameters': hyperparameters  # Included for future reference
}

# Saving the checkpoint to the specified path
torch.save(checkpoint, checkpoint_path)

print("Checkpoint saved successfully.")

print("Training complete. Model saved as checkpoint.")

