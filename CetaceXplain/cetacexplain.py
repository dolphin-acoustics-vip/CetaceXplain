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

    def get_label_indices(self, test_data, multiclass_test_labels):

        """This function returns the indices of labels for predicted labels and actual labels. 
        The label indices should correspond to self.class_names.
        

        Parameters
        ----------
        test_data : numpy array
            Array of data for testing
        multiclass_test_labels : numpy array
            Correct labels corresponding to test_data arrays

        Returns
        -------
        tuple
            Tuple of arrays with test label indices and predicted label indices (in that order)

        """

        probability_scores = self.model.predict(test_data)  # list of predicted class probabilities for each image
                                                            # model.predict better for batches but equivalent to model()
        predicted_label_indices = probability_scores.argmax(axis = 1)
        test_label_indices = multiclass_test_labels.argmax(axis = 1)
        return test_label_indices, predicted_label_indices 
    
    def report_stats(self, test_data, multiclass_test_labels):
        """This function returns statistical metrics about the model's performance, 
        including precisiion, recall and F1 score.
        

        Parameters
        ----------
        test_data : numpy array
            Array of data for testing
        multiclass_test_labels : numpy array
            Correct labels corresponding to test_data arrays

        Returns
        -------
        string
            String giving statistical metrics on the model's performance

        """
        # get indices of labels for predicted labels and actual labels 
        test_label_indices, predicted_label_indices = self.get_label_indices(test_data, multiclass_test_labels)

        return classification_report(test_label_indices, predicted_label_indices, target_names = self.class_names, 
                                    labels = np.unique(np.concatenate([predicted_label_indices, test_label_indices])))

    def show_confusion_matrix(self, test_data, multiclass_test_labels):

        """
        This function plots a confusion matrix for some given data by comparing the 
        model's predictions with provided test labels.

        Parameters
        ----------
        test_data : numpy array
            Array of data of for testing
        multiclass_test_labels : numpy array
            Correct labels corresponding to test_data array

        Returns
        -------
        None.

        """

        # get indices of labels for predicted labels and actual labels 
        test_label_indices, predicted_label_indices = self.get_label_indices(test_data, multiclass_test_labels)
        
        # adapted from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay.from_predictions
        # create confusion matrix object and plot its results
        cm = confusion_matrix(test_label_indices, predicted_label_indices, 
                                labels = np.unique(np.concatenate([predicted_label_indices, test_label_indices])))
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

    def explain(self, test_images , multiclass_test_labels = None, background_images = None, 
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
        multiclass_test_labels : numpy array
            Correct image labels corresponding to test_images arrays
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

        # print predicted labels and actual labels for each test image
        test_label_indices, predicted_label_indices = self.get_label_indices(test_images, multiclass_test_labels)
        print("Predicted Labels\t\tActual Labels\t\t(in order of appearance of images below)")
        for i in range(len(predicted_label_indices)):
            print(self.class_names[predicted_label_indices[i]] + "\t\t" + self.class_names[test_label_indices[i]])

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

