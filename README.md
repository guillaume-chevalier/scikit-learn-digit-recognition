# Digit Recognition with a scikit-learn's Bernoulli RBM and a Logistic Classifier.


## INSTALL REQUIREMENTS:

1. Enthought Canopy
2. scikit-learn
3. iPython
4. openCV
The individual instructions are found in the "REQUIREMENTS" folder. 


## DESCRIPTION:

The final program reads pictures from your machine's webcam and tries to identify what number is seen. 
The number should be centred and occupying the whole webcam's field of view to be accurately recognised. 
A short time after launching the program, a window will appear to show the prediction's statistic and
the image in grayscale 16x16 px that the neural network recieved. 

The technology used behind this is a classifier containing a Bernoulli RBM (a kind of neural network) 
which is followed by a logistic classifier. 
To further improve this program, the classifier should learn to recognise a wider and noisier dataset, 
and should learn to recognise what is not a number to be able to pool multiple translated and scaled 
versions of the input image. What is commonly called a ConvNet could have been used too. 

