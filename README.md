### big-data-technology-project

Project Goal
CNN to solve Sudoku puzzles

##Summary

As a part of a team project, I created a training data set for our convolutional neural network, CNN, model. The model input is an image of an unsolved Sudoku puzzle, the model extracts the cells with prefilled numbers, predicts the numbers present, and based on the predictions solves the Sudoku puzzle. 

I created 37,800 labeled images with printed digits with 4,200 images of each digit 1-9. The training data set has 60 different fonts and was created using InDesign and Python. The original 540 images were exported in order to maintain the order of the images in the file directory. With this predictability, the corresponding labels were added to a numpy file in Python. 

The original ‘clean’ images were then transformed 70 times using numpy transformations and overlaying images using openCV in Python. The additional corresponding labels were created alongside the transformations. The final training data set had printed numbers with borders and noise to more accurately reflect the numbers extracted from the Sudoku puzzles.

Results
With the printed digits training data set, we achieved 96.2% accuracy rather than 23.1% accuracy with the well known handwritten MNIST data set. Note the use of a convolutional neural network was a project requirement. OCR, optical character recognition, is a potentially a more reliable approach within our Sudoku project than the CNN model.

Included
Images of process and model results

Programs
Python, Adobe InDesign

Packages
OpenCV, numpy, itertools, matplotlib

Team
Richard Gower, Bryon Mosier, Joshua Roach, Veronica Stephens, Donald Villarreal, Aaron Wright

Course: Information Systems Big Data Technology
Programs: Jupyter, Python, Spark (cloud-based environments)
Course Topics: Introduction to Big Data, Big Data Characteristics, Hadoop Ecosystems, Big Data Integration, Big Data Processing, Graph Analytics for Big Data, Machine Learning for Big Data, Big Data Visualization
