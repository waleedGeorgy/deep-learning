import torchvision
import torch
from torch import nn

def create_vitb16(num_class:int = 3, seed:int = 42):
  """
  A class to create a ViT-B/16 feature extractor with DEFAULT weights

  Args:
    num_class (int): number of classes in the classifier head (int).
    seed (int): random seed values (int, default = 42)

  Returns:
    model (torch.nn.Module): an instance of the ViT-B/16 feature extractor with DEFAULT weights.
    transforms (torchvision.transforms): an instant of the ViT-B/16 transforms
  """
  weights = torchvision.models.ViT_B_16_Weights.DEFAULT
  transforms = weights.transforms()
  model = torchvision.models.vit_b_16(weights = weights)

  for param in model.parameters():
    param.requires_grad = False

  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)

  model.heads = nn.Sequential(nn.Linear(in_features=768,
                                          out_features=num_class))

  return model, transforms
