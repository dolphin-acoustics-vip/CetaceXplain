# CetaceXplain
This is a module meant to provide useful ways of visualizing and understanding the  manner in which machine learning models classify cetacean whistle contours. One of the most useful features of this module is that it uses SHAP (https://github.com/slundberg/shap) among other to
 highlight pixels of importance to machine learning models' decisions.
 
## Example of Use
Below are two examples of the SHAP image explanations, with the terminal output above showing the corresponding predicted labels, and actual (correct) labels. They were taken with a different number of background images over which SHAP was calculated (SHAP needs some background images beforehand to do the explanations). Roughly speaking, the more the background images, the more comprehensive the pixel highlighting.
<img src = "images/Sample_Image_Explanation1.png" width = "700">
