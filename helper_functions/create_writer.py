from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os

# Creating a function to create custom SummaryWriter objects to track experiments
def create_writer(model_name: str,
                  experiment_name: str,
                  extra: str):
  """
  Creates a torch.utils.tensorboard.SummaryWriter to track model experiments.
  Saves model's tracked parameters in a path of form 'runs/timestamp/model_name/
  experiment_name/extra' if 'extra' was defined by the user.

  Args:
    model_name: a string representing the name of the model to be tracked.
    experiment_name: a string the defines the name of the experiment to be performed on the model.
    extra: a string containig any additional info the user would like to add to the writer.
  
  Returns:
    A SummaryWriter object that saves model's logs in the directory 'runs/timestamp/model_name/
    experiment_name/extra' if 'extra' was defined by the user.
    Or in the directory runs/timestamp/model_name/ experiment_name/ otherwise.
  
  Example Usage:
    create_writer(model = 'effnetb2_model',
    experiment_name = 'foodVision_10_percent_data',
    extra = '5_epochs')
    This will create a log directory in 'runs/CURRENT_YEAR_MONTH_DAY/effnetb2_model/
    foodVision_10_percent_data/5_epochs'
  """
  # Get the current datatime in format YYYY/mm/dd
  timestamp = datetime.now().strftime('%Y-%m-%d')

  # Creating log_dir path
  if extra:
    log_dir = os.path.join('runs', timestamp, model_name, experiment_name, extra)
  else:
    log_dir = os.path.join('runs', timestamp, model_name, experiment_name)
  
  print(f'Creating a log directory in {log_dir}')
  return SummaryWriter(log_dir = log_dir)
