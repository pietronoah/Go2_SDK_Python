import yaml
import torch
import torch.nn as nn

# Function to load YAML configuration
def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Function to load the actor network
def load_actor_network(config, model_path='nn/model.pt'):
    model = torch.jit.load(model_path)
    model.eval()
    return model