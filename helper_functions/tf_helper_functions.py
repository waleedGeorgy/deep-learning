import os
import tensorflow as tf
import zipfile
import datetime
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def dir_walk(dir):
  '''
  Walks through a directory and displays all the sub-directories and files within it.

  Args:
    dir: the directory to be walked through.
  '''
  for dirpath, dirnames, filenames in os.walk(dir):
    print(f'There are {len(dirnames)} directories and {len(filenames)} files in {dirpath}')

def load_and_prep_img(img_path:str,
                      img_size:int = 224,
                      rescale:bool = False):
  '''
  Reads in an image and turns it into a tensor.

  Args:
    img_path (str): the path of the image.
    img_size (int): the size which the image will be reshaped into (default = 224).
    rescale (bool): whether the image needs to be normalized or not.
  Returns:
    The image as a normalized tensor.
  '''
  # Reading and decoding the image into a tensor
  img = tf.io.read_file(img_path)
  img = tf.image.decode_image(img)
  # Resiszing the image into a default image size
  img = tf.image.resize(img, [img_size, img_size])
  # Normalizing the image
  if rescale:
    img = img/255.
  else:
    img
  return img

def plot_class_report_and_confmat(y_true, y_pred, class_names, savefig = False):
  '''
  Plots the classification report and the confusion matrix of a TensorFlow model.

  Args:
    y_true: the true labels of the data.
    y_pred: the predicted labels of the data obtained from a model.
    class_names: the names of the classes in the classificaiton problem.
    savefig: when set to "True" saves the confusion matrix as a .png figure.
  '''
  # Printing the classification report
  print(classification_report(y_true, y_pred, target_names = class_names))

  # Plotting the confusion matrix
  cm = confusion_matrix(y_true, y_pred)
  confmat = ConfusionMatrixDisplay(cm, display_labels = class_names)
  confmat.plot()

  # Saving the confusion matrix as a figure
  if savefig == True:
    confmat.figure_.savefig('confusion_matrix.png')

def unzip_data(path:str):
  '''
  Unzips a .rar file into the current working directory.

  Args:
    path (str): the path of the file to be unzipped.
  '''
  print('Unzipping data...')
  with zipfile.ZipFile(path) as zipref:
    zipref.extractall()
  print('Done!')

def pred_and_plot(model, filename, class_names, img_size = 224, rescale = False):
  '''
  Reads in an image, turns it into a tensor, resizes it into a default
  size of (224x224x3), and rescales it if neccessary. Then predicts
  the class of the image using a trained TensorFlow model and plots it
  along with its predicted class, true class, and maximum prediction
  probability.

  Args:
    model: the trained TensorFlow model that will make the prediction.
    filename: the path of the image that needs to be predicted on.
    class_names: the names of the classes the model was trained on.
    img_size (default = 224): the height and width the image will be resized into.
    rescale (default = False): If normalization should be applied to the image or not.
  '''
  # Turning the image into a tensor and adding the batch dimension to it
  img = tf.io.read_file(filename)
  img = tf.image.decode_image(img)
  # Resiszing the image into a default image size
  img = tf.image.resize(img, [img_size, img_size])
  # Normalizing the image
  if rescale:
    img = img/255.

  # Adding the batch dimension
  img = tf.expand_dims(img, axis = 0)

  # Making a prediction on the image using the model
  pred_prob = model.predict(img)[0]

  # Checking if the classification is binary or multi-class and applying the appropriate functions
  if len(pred_prob) > 1:
    pred_class = class_names[tf.argmax(pred_prob)] 
  else:
    pred_class = class_names[int(tf.round(pred_prob)[0])]
  # Getting the true class of the image
  true_class = filename.split('/')[-2]
  
  # Plotting the image and its predicted class
  if pred_class == true_class:
    text_color = 'g'
  else:
    text_color = 'r'
  plt.imshow(tf.squeeze(img/255.))
  plt.title(f'True class: {true_class} | Predicted class: {pred_class} | Prediction probability: {max(pred_prob):.4f}', c = text_color)
  plt.axis(False);

def plot_curves(history):
  '''
  Plots the training and validation loss and accuracy curves of a trained model.

  Args:
    history: the history of a TensorFlow model.
  '''
  # Getting the range of epochs for x-axis
  epochs = range(len(history.history['loss']))

  # Getting the loss and accuracy curves for the y-axis
  loss = history.history['loss']
  acc = history.history['accuracy']
  val_loss = history.history['val_loss']
  val_acc = history.history['val_accuracy']

  # Plotting the loss curves
  plt.figure()
  plt.plot(epochs, loss, label = 'Training Loss')
  plt.plot(epochs, val_loss, label = 'Validation Loss')
  plt.legend()
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Loss Curves')

  # Plotting the accuracy curves
  plt.figure()
  plt.plot(epochs, acc, label = 'Training Accuracy')
  plt.plot(epochs, val_acc, label = 'Validation Accuracy')
  plt.legend()
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.title('Accuracy Curves');

def create_tensorboard_callback(dir_name:str, experiment_name:str):
  '''
  Creates a TensorBoard callback in the "dir_name/experiment_name/current_datetime/"
  directory.

  Args:
    dir_name: target directory to store TensorBoard log files
    experiment_name: name of experiment directory (e.g. efficientnet_model_1)
  
  Returns:
    TensorBoard callback
  '''
  # Getting the current date and time
  current_datetime = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

  # Creating the directory where the TensorBoard log will be saved
  log_dir = dir_name + "/" + experiment_name + "/" + current_datetime

  # Creating the callback
  callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  print(f"Saving TensorBoard log files to: {log_dir}")

  return callback
  
def compare_finetune_history(original_history, new_history, initial_epochs=5):
    """
    Compares two model history objects. Shows the loss and accuracy curves before
    and after fine-tuning.

    Args:
      original_history: history object of a model before fine-tuning.
      new_history: history object of a model after fine-tuning.
      initial_epochs: the epoch where fine-tuning starts.
    """
    # Get original history metrics
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]
    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]
    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Get the range of epochs for the x-axis
    epochs = range(len(total_loss))

    # Show accuracy plots
    plt.figure(figsize=(7, 7))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, total_acc, label = 'Training Accuracy')
    plt.plot(epochs, total_val_acc, label = 'Validation Accuracy')
    plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label = 'Start of Fine Tuning')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    # Show loss plots
    plt.subplot(2, 1, 2)
    plt.plot(epochs, total_loss, label = 'Training Loss')
    plt.plot(epochs, total_val_loss, label = 'Validation Loss')
    plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label = 'Start of Fine Tuning')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch');
