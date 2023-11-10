import argparse
from torchvision import transforms, models
import torch
import json
from utils import load_checkpoint, process_image, imshow, predict, display_prediction

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, help='Path to the input image')
parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file')
parser.add_argument('--top_k', type=int, default=3, help='Return top K most likely classes')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to category names mapping file')
parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
args = parser.parse_args()

# Check if GPU is available
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

# Load the checkpoint
model, class_to_idx = load_checkpoint(args.checkpoint)


# Load category names mapping

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
print(len(cat_to_name))


# Display the results
display_prediction(args.image_path, model, cat_to_name, args.top_k)

