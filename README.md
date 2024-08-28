# fer2013: Facial Expression Recognition with Pytorch

Using a convolutional neural network, achieved 63% accuracy (human performance is 65%, state of the art is 73%)
  
Loosely based on the model structure described in "Facial Emotion Recognition: State of the Art Performance on FER2013" by Yousif Khaireddin and Zhuofa Chen (https://arxiv.org/pdf/2105.03588)

## Requirements

fer2013 dataset from https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data  
  
Python: 3.12.5  
  
Torch: 2.4.0  
  
Torchvision: 0.19.0  

## Usage

Ensure fer2013 files from Kaggle are in the repository, in a folder named "fer2023"  
  
To demo trained model: ```python model_demo.py```  
  
To retrain model: ```python model_train.py``` 
  
To demo retrained model: change ```path``` variable of model_demo.py to ```'fer2013_new.pth'```
