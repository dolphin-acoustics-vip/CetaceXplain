from sklearn.metrics import classification_report
import numpy as np
import shap
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

tf.compat.v1.disable_v2_behavior() # disabling certain newer tf features for compatibility with shap

# TODO: Neaten code (remove/fix commented out code)
# TODO: Add method for loading models? 
# TODO: Figure out why explain function is so memory intensive
# TODO: Add labels to shap explainer inage
# TODO: Figure out what image_shape is supposed to be
# TODO: Look at https://www.tensorflow.org/api_docs/python/tf/compat/v1/keras/experimental/export_saved_model (although depracated)
# TODO: Write proper docstring comments, and declare variable type where useful (see https://docs.python.org/3/library/typing.html)

# create super class for more general model to which other SHAP methods may be applied
class ClassifierModel:
  """ This class is responsible for creating a dolphin acoustics classifier model.
  """

  def __init__(self, model):
    self.model = model
    self.class_names = None

  def report_stats(self, test_data, test_labels):
    """ This function returns statistical metrics about the model's preformance, 
        including precisiion, recall and F1 score.
    """

    probability_scores = self.model.predict(test_data)  # list of predicted class probabilities for each image
                                                        # model.predict better for batches but equivalent to model()
    predicted_labels = probability_scores.argmax(axis = 1)
    test_labels = test_labels.argmax(axis = 1)

    return classification_report(test_labels, predicted_labels, target_names = self.class_names, 
                                 labels = np.unique(np.concatenate([predicted_labels, test_labels])))

  def show_confusion_matrix(self, test_data, test_labels):
    """ This function plots a confusion matrix for some given data by comparing the model's prediction with provided test labels.  
    """
    probability_scores = self.model.predict(test_data)  # list of predicted class probabilities for each image
                                                        # model.predict better for batches but equivalent to model()
    predicted_labels = probability_scores.argmax(axis = 1)
    test_labels = test_labels.argmax(axis = 1)

    # adapted from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay.from_predictions
    cm = confusion_matrix(test_labels, predicted_labels, 
                          labels = np.unique(np.concatenate([predicted_labels, test_labels])))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                     display_labels=self.class_names)
    display.plot()


class ImageClassifierModel(ClassifierModel):
  """ This class is responsible for creating an image classifier model.
  """

  def __init__(self, model, data_labels, **kwargs):
    
    super().__init__(model)
    self.image_data = kwargs.get("image_data", None) # second argument is default value
    self.data_labels = data_labels
    self.class_names = kwargs.get("class_names", list(set(data_labels)))

  def classify(self, x):
    return model(x) # model.predict model(np.array(x))
  
  # TODO: Remove or fix commented out code section
  """def explain(self, image_data = None, **kwargs):
    
    # set image data for explanation as the same which was 
    # used for the model, if nothing else is specified
    if image_data is None:
      image_data = self.image_data

    batch_size = kwargs.get("batch_size", 50)
    outputs = kwargs.get("outputs", shap.Explanation.argsort.flip[:2]) #4 gets 2 most probable images
    start_image_index = kwargs.get("start_image_index", 1) # index of first image to be explained
    end_image_index = kwargs.get("end_image_index", 3) # index following last image to be explained
    max_evals = kwargs.get("max_evals", 100)
    
    # define a masker that is used to mask out partitions of the input image.
    masker = shap.maskers.Image("inpaint_telea", self.image_data[0].shape)

    # create an explainer with model and image masker
    explainer = shap.Explainer(self.classify, masker, output_names=self.class_names) 
 
    # here we explain two images using 100 evaluations of the underlying model to estimate the SHAP values
    #explained_images = np.array([self.image_data[0][0][x] for x in range(start_image_index,end_image_index)])
    shap_values = explainer(self.image_data[start_image_index:end_image_index], max_evals=max_evals, 
                            batch_size=batch_size) # batch_size=batch_size # outputs=outputs

    print("shap values shape: ", shap_values[0].shape[0])
    print(shap_values.output_dims)
    shap.image_plot(shap_values)"""

  def explain(self, train_images, image_limit = 5):
    """ This function displays an image highlighting relevant pixels to the model's classification decision.
    """
    # TODO: Add commented out code for labels (removed for now to reduce computational strain)
    # TODO: Remove magic constants in shap_values() and image_plot()

    # adapted from https://github.com/slundberg/shap#deep-learning-example-with-deepexplainer-tensorflowkeras-models
    
    # select a set of background examples to take an expectation over
    background = train_images[0:image_limit]#[np.random.choice(train_images.shape[0], image_limit, replace=False)]

    # explain predictions of the model on four images
    e = shap.DeepExplainer(self.model, background)

    shap_values = e.shap_values(train_images[1:4])

    
    """label_indices = [np.argmax(self.model.predict(np.array([image]))) for image in train_images[1:4]]
    labels = np.array([self.class_names[index] for index in label_indices])
    labels = labels.reshape(labels.shape[0], 1)"""

    # plot the feature attributions
    shap.image_plot(shap_values, train_images[1:4]) #TODO: add class names

  def show_image(image_array_data):
    plt.imshow(image_array_data)

