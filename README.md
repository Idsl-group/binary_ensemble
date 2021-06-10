# Binary Ensemble

This is an ensemble model of binary classification neural networks.

The biggest problem with thermal comfort datasets is that they are often imbalanced. When self-reporting, people are much more likely to report that they are comfortable as opposed to too warm or too cold. This creates a major class imbalance because now, most of the dataset has been labelled "comfortable" and training becomes much harder for machine learning models. The below graphic shows the class imbalance in one particular dataset:
![](https://i.imgur.com/SG6QDFB.png)

There are multiple ways to combat this such as undersampling the majority class (however, this gets rid of precious data) or synthesizing samples for the minority classes using techniques such as SMOTE, ADASYN, or even using neural networks such as GANs. 

This repo proposes a new method to combat imbalanced datasets by creating 'k' binary classifiers for each class (and combining the non target classes into one class for each model), synthesizing data to combat the remaining imbalance using a CGAN, and then classifying by using the maximum probability across all the classifiers for each class. A workflow for this is shown below:
![](https://i.imgur.com/8OAUCFb.png)
This image shows the k classifiers for one particular class, and we would create a new ensemble for each class in the dataset. 

This model would work only when the imbalance ratio between the majority class and minority classes is high, while the imbalance ratio between the minority classes themselves is low, which is exactly what the imbalance in thermal comfort datasets look like. 

