import torchvision
import torch
from torch import nn

def create_effnetb2(num_class: int,
                    seed: int = 42):
  """
  A class to create an EfficientNetB2 feature extractor with DEFAULT weights

  Args:
    num_class (int): number of classes in the classifier head (int).
    seed (int): random seed values (int, default = 42)

  Returns:
    model (torch.nn.Module): an instance of the EfficientNetB2 feature extractor with DEFAULT weights.
    transforms (torchvision.transforms): an instant of the EfficientNetB2 transforms
  """
  weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
  transforms = weights.transforms()
  model = torchvision.models.efficientnet_b2(weights = weights)

  for param in model.features.parameters():
    param.requires_grad = False

  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)

  model.classifier = nn.Sequential(
      nn.Dropout(p = 0.3, inplace = True),
      nn.Linear(in_features = 1408, out_features = num_class, bias = True)
  )

  return model, transforms
