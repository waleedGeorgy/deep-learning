from pathlib import Path
import torch
from torch import nn
import os

# Function to save models
def save_model(model: torch.nn.Module,
               target_path: str,
               model_name:str):
  """
  Saves a model to a user-defined target path

  Args:
    model: a torch.nn.Module model that needs to be saved
    target_path: a string of where the model will be saved to.
    model_name: a string defining the name of the model to be saved (name of model
    must end with .pt or .pth).
  """
  
  main_models_path = Path(target_path)
  main_models_path.mkdir(parents = True, exist_ok = True)

  if model_name.endswith('.pt') or model_name.endswith('.pth'):
    model_path = main_models_path / model_name
    print(f'[INFO] Saving model to {model_path}')
    torch.save(obj=model.state_dict(), f= model_path)
  else:
    print('[ERROR] model_name must have the extension .pt or .pth')
