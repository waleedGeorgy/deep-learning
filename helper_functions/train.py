from typing import Dict, List, Tuple
from tqdm.auto import tqdm
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# Create manual seeds
def set_seeds(seed:int = 42):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)

# Setting-up our code to be device agnostic
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Train step function -------------------
def train_step(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = device) -> Tuple[float, float]:
  """
  Performs a single train step on the PyTorch model for one epoch.

  Args:
    model: a torch.nn model that will be trained.
    train_dataloader: a torch.utils.data.DataLoader that contain the batched train data.
    loss_fn: a torch.nn loss function that quantifies how wrong the model is.
    optimizer: a torch.optim optimizer that will update the weights to minimize the loss function.
    device: a torch.device target device that will perform the computations.

  Returns:
    A tuple that contains the train loss and train accuracy values in the form Tuple(train_loss, train_accuracy)
  """
  # Putting the model into train mode
  model.train()

  # Initializing the loss and accuracy parameters
  train_loss, train_acc = 0, 0

  # Looping through train dataloader and performing standard model training steps
  for batch, (X, y) in enumerate(train_dataloader):
    X, y = X.to(device), y.to(device)

    y_logits = model(X)
    loss = loss_fn(y_logits, y)
    train_loss += loss.item()

    y_preds = torch.softmax(y_logits, dim = 1).argmax(dim = 1)
    train_acc += (y_preds == y).sum().item() / len(y_preds)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  # Calculating the value of loss and accuracy parameters
  train_loss /= len(train_dataloader)
  train_acc /= len(train_dataloader)

  return train_loss, train_acc

# Test step function --------------
def test_step(model: torch.nn.Module,
              test_dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device = device) -> Tuple[float, float]:
  """
  Performs a single test step on the PyTorch model for one epoch.

  Args:
    model: a torch.nn model that will be trained.
    test_dataloader: a torch.utils.data.DataLoader that contain the batched test data.
    loss_fn: a torch.nn loss function that quantifies how wrong the model is.
    device: a torch.device target device that will perform the computations.

  Returns:
    A tuple that contains the test loss and test accuracy values in the form of Tuple(test_loss, test_accuracy)
  """

  # Initializing the test loss and accuracy parameters
  test_loss, test_acc = 0, 0

  # Putting the model into evaluation mode and opening up the inference mode context manager
  model.eval()
  with torch.inference_mode():

    # Looping through the test dataloader and performing standard model testing steps
    for X_test, y_test in test_dataloader:
      X_test, y_test = X_test.to(device), y_test.to(device)

      test_logits = model(X_test)
      test_loss += loss_fn(test_logits, y_test).item()

      test_preds = torch.softmax(test_logits, dim = 1).argmax(dim = 1)
      test_acc += (test_preds == y_test).sum().item() / len(test_preds)

    # Calculating the value of test loss and accuracy parameters
    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)

  return test_loss, test_acc

# Main train function that leverages the train_step and test_step functions
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          writer: torch.utils.tensorboard.writer.SummaryWriter = None,
          seed: int = None,
          device: torch.device = device) -> Dict[str, List]:
  """
  Trains and tests a PyTorch model for user-defined number of epochs, and tracks
  the loss and accuracy of the model across epochs.

  This functions leverages the train_step and test_step functions in order to
  train and test the model. The writer to track the model's results can be instantiated
  using the create_writer() function.

  Args:
    model: a torch.nn model that will be trained.
    train_dataloader: a torch.utils.data.DataLoader that contain the batched train data.
    test_dataloader: a torch.utils.data.DataLoader that contain the batched test data.
    loss_fn: a torch.nn loss function that quantifies how wrong the model is.
    optimizer: a torch.optim optimizer that will update the weights to minimize the loss function.
    epochs: an integer that decides how many epochs the model will be trained and tested for.
    writer: if defined, create a torch.utils.tensorboard.writer.SummaryWriter to track model's results.
    seed: an integer, if defined, sets the manual seed.
    device: a torch.device target device that will perform the computations.

  Returns:
    A dictionary that contains the train loss and accuracy, and test loss and accuracy.
    In the form of: {'train_loss':[], 'train_acc':[], 'test_loss':[], 'test_acc':[]}.
    for example: when training for 2 epochs
                  {'train_loss':[0.6523, 0.4219],
                  'train_acc':[0.672, 0.812],
                  'test_loss':[0.7771, 0.5326],
                  'test_acc':[0.727, 0.991]}
  """
  # Create a manual seed
  if seed:
    set_seeds(seed)

  # Initializing the dictionary that will store the values of the training and testing
  results = {'train_loss':[],
             'train_acc':[],
             'test_loss':[],
             'test_acc':[]
             }

  # Sending the model to the target device
  model.to(device)

  # Performing the train and test steps equal to the number of epochs
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model = model,
                                      train_dataloader=train_dataloader,
                                      loss_fn = loss_fn,
                                      optimizer = optimizer,
                                      device = device)


    test_loss, test_acc = test_step(model = model,
                                    test_dataloader = test_dataloader,
                                    loss_fn = loss_fn,
                                    device = device)

    # Appending the loss and accuracy values to the dictionary
    results['train_loss'].append(train_loss)
    results['train_acc'].append(train_acc)
    results['test_loss'].append(test_loss)
    results['test_acc'].append(test_acc)

    # Tracking models' results using SummaryWriter()
    if writer:

      # Adding results to be tracked to the writer
      writer.add_scalars(main_tag = 'Loss', tag_scalar_dict = {'train_loss': train_loss, 'test_loss': test_loss},
                         global_step = epoch)
      writer.add_scalars(main_tag = 'Accuracy', tag_scalar_dict = {'train_acc': train_acc, 'test_acc': test_acc},
                         global_step = epoch)
      writer.add_graph(model=model, input_to_model=torch.randn(32, 3, 224, 224).to(device))
      # Closing the writer
      writer.close()

    # If no writer defined, skip writer creation
    else:
      pass

    # Printing out the results
    print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.3f} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.3f}')

  return results
