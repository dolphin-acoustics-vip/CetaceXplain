from sklearn.metrics import classification_report
import numpy as np
import shap
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

tf.compat.v1.disable_v2_behavior() # disabling certain newer tf features for compatibility with shap
 
# TODO: Find a way to use shap values to recreate audible (wavfile) version of what highlighted spectrograms indicate
# TODO: Figure out why explain function is so memory intensive
# NOTE: Loading transfer learning models with the new tensorflow folder saving format seems to give an error for model loading,
#       so it is safer to save them in an h5 format for now. Custom-made models seem to be fine with being loaded in 
#       the newer format though. 

# create super class for more general model to which other SHAP methods may be applied
class ClassifierModel:
    """ This class is responsible for creating a dolphin acoustics classifier model.
    """

    def __init__(self, model):
        self.model = model
        self.class_names = None

    def report_stats(self, test_data, test_labels):
        """This function returns statistical metrics about the model's performance, 
        including precisiion, recall and F1 score.
        

        Parameters
        ----------
        test_data : numpy array
            Array of data for testing
        test_labels : numpy array
            Correct labels corresponding to test_data image arrays

        Returns
        -------
        string
            String giving statistical metrics on the model's performance

        """

        probability_scores = self.model.predict(test_data)  # list of predicted class probabilities for each image
                                                            # model.predict better for batches but equivalent to model()
        predicted_labels = probability_scores.argmax(axis = 1)
        test_labels = test_labels.argmax(axis = 1)

        return classification_report(test_labels, predicted_labels, target_names = self.class_names, 
                                    labels = np.unique(np.concatenate([predicted_labels, test_labels])))

    def show_confusion_matrix(self, test_data, test_labels):

        """
        This function plots a confusion matrix for some given data by comparing the 
        model's predictions with provided test labels.

        Parameters
        ----------
        test_data : numpy array
            Array of data of for testing
        test_labels : numpy array
            Correct labels corresponding to test_data image arrays

        Returns
        -------
        None.

        """

        probability_scores = self.model.predict(test_data)  # list of predicted class probabilities for each image
                                                                # model.predict better for batches but equivalent to model()
        predicted_labels = probability_scores.argmax(axis = 1)
        test_labels = test_labels.argmax(axis = 1)
        
        # adapted from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay.from_predictions
        # create confusion matrix object and plot its results
        cm = confusion_matrix(test_labels, predicted_labels, 
                                labels = np.unique(np.concatenate([predicted_labels, test_labels])))
        display = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                            display_labels=self.class_names)
        display.plot()

# create image-related subclass extending ClassifierModel class
class ImageClassifierModel(ClassifierModel):

    """This class is responsible for creating an image classifier model.
    """

    def __init__(self, model, image_data, data_labels, **kwargs):
        """
        Parameters
        ----------
        model : tensorflow model
            Machine learning model for classification
        image_data : numpy array
            Array of image arrays of for testing
        data_labels : list
            Labels corresponding to the class of each image
        data_labels : numpy array
            Correct labels corresponding to test_data image arrays
        class_names : list, optional
            Set of (unique) image labels, deduced 
            from unique values of data_labels if not specified.
        
        Returns
        -------
        None.

        """
        
        super().__init__(model)
        self.image_data = image_data
        self.data_labels = data_labels
        
        # get unique label names if not directly specified
        self.class_names = kwargs.get("class_names", list(set(data_labels))) 

    def explain(self, test_images , background_images = None, 
        background_image_limit: int = 20, test_image_limit: int = 4):

        """
        This function displays an image highlighting relevant pixels to the model'shap 
        compatibility  classification decision.
        
        Note: This function can be quite computationally expensive 
        depending on the choice of machine learning model. 
        It is advised that the number of background images be chosen carefully, 
        with a low number (perhaps 20 or less) being safer to start with if unsure. 

        Parameters
        ----------
        test_images : numpy array
            Array of image arrays of for testing
        background_images : numpy array, optional
            Images used to get background understanding of the model's expected predictions.
            The default is self.image_data.
        background_image_limit : int, optional
            Maximum number of background images used in calculation.
            The default is 20.
        test_image_limit : int, optional
            Maximum number of test images to be explained, starting at index 0, 
            and ending at index before test_image_limit. The default is 4.

        Returns
        -------
        None.

        """
        
        if background_images is None:
            background_images = self.image_data

        # adapted from https://github.com/slundberg/shap#deep-learning-example-with-deepexplainer-tensorflowkeras-models
        
        # select a set of background examples to take an expectation over
        if len(background_images) > background_image_limit:
            background_images = background_images[0 : background_image_limit] #[np.random.choice(train_images.shape[0], background_image_limit, replace=False)]
        
        # select test images to be explained
        if len(test_images) > test_image_limit:
            test_images = test_images[0 : test_image_limit]
    
        # explain predictions of the model
        explainer = shap.DeepExplainer(self.model, background_images)
    
        # get shap values from explainer
        shap_values = explainer.shap_values(test_images)
    
        # create labels for image plots
        labels = np.array(self.class_names)
        labels = np.tile(labels, shap_values[0].shape[0]) # duplicate labels for every row of images
        labels = np.reshape(labels, (shap_values[0].shape[0], len(shap_values))) # reshape array appropriately 
                                                                                 # for shap compatibility 

        # print predicted labels for each test image
        probability_scores = self.model.predict(test_images)  
        predicted_label_indices = probability_scores.argmax(axis = 1)
        print("Predicted Labels (in order of appearance of images below):")
        for index in predicted_label_indices:
            print(self.class_names[index])

        # plot the image explanations
        shap.image_plot(shap_values, test_images, labels = labels)

def show_image(image_array_data):
      """
      Display image from numpy array image representation

      Parameters
      ----------
      image_array_data : numpy array
          array image representation

      Returns
      -------
      None.

      """
      plt.imshow(image_array_data)

def convert_time_series_to_wavfile():
      """This funcion converts a time series of frequency values to a wave file
      """
      pass

