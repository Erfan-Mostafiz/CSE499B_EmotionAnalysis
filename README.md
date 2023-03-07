# Real-Time Emotion Recognition from Frontal Face Videos using Deep Learning Techniques

This repository contains all the source code of our final semester thesis research project of real-time emotion detection. </br>
Used Deep Learning Techniques and our CNN based model achieved a `Test Accuracy` of `75.4%` on the FER2013 dataset.

### The final thesis paper that talks about our research can be found in [this link](https://drive.google.com/file/d/1-IyRQajMiPwk1ABdz2CgN1807POL7c0P/view?usp=sharing)

## Emotion Detection Process
<p align = "left">
  The following Image illustrates the Real Time Emotion Detection Process of our system: 
</p>
<p align="center">
  <img  src="https://github.com/Erfan-Mostafiz/CSE499B_EmotionAnalysis/blob/main/RealTime%20Design.png?raw=true">
</p>

1) A person's face is first detected from a video using `Haarcascade Frontal Face Detector`. 
2) This detects the face and converts each frames of image into `greyscale images`. 
3) These frames are then passed onto the trained emotion detector AI model, which predicts the emotion of the person in that frame.
4) The emotions are then returned onto the front-end.
5) A `bounding box` is drawn around the detected face.
6) The detected emotion is then shown above the bounding box (bBox).
7) All this happens in `real-time`, i.e. the face in each frame is detected, emotion predicted and shown over the bBox before the next frame emotion is predicted.
8) Used `OpenCV-python` to handle the video captured via camera, draw bBox and the predicted emotion over the bBox.

## CNN Model Architecture & Hyperparameters used
The Diagram below illustrates our CNN based network architecture:
<p align="center">
  <img src="https://github.com/Erfan-Mostafiz/CSE499B_EmotionAnalysis/blob/main/Realtime%20Detection%20CNN%20based%20Network/CNN%20Architecture.png" width="1000" height ="300" />
  
The Table below shows the Hyperparameters used in our CNN based network:
| First Header  | Second Header |
| ------------- | ------------- |
| Loss Function  | Categorical Cross Entropy  |
| Optimizer | Adam  |
| Epochs  | 100  |
| Metric  | Accuracy  |
| Learning Rate  | 0.001  |
| Batch Size  | 64  |

## Real-Time Inference 
<p align = "left">
  The model has been tested real-time on one of the author's face, and the results are shown below: 
</p>
<p float="center">
  <img src="https://github.com/Erfan-Mostafiz/CSE499B_EmotionAnalysis/blob/main/Inference_Real%20Time/One%20Author's%20Real%20Time%20Emotion/Happy.png?raw=true" width="240" height ="250" />
  <img src="https://github.com/Erfan-Mostafiz/CSE499B_EmotionAnalysis/blob/main/Inference_Real%20Time/One%20Author's%20Real%20Time%20Emotion/Angry.png?raw=true" width="240" height ="250" /> 
  <img src="https://github.com/Erfan-Mostafiz/CSE499B_EmotionAnalysis/blob/main/Inference_Real%20Time/One%20Author's%20Real%20Time%20Emotion/Neutral.png?raw=true" width="240" height ="250" />
  <img src="https://github.com/Erfan-Mostafiz/CSE499B_EmotionAnalysis/blob/main/Inference_Real%20Time/One%20Author's%20Real%20Time%20Emotion/Surprise.png?raw=true" width="240" height ="250" />
</p>


## Test Result Analysis
The Confusion Matrix of our CNN model shows how accurate this model is in predicting the emotions:
<p align="left">
  <img src="https://github.com/Erfan-Mostafiz/CSE499B_EmotionAnalysis/blob/main/Test%20Result/Confusion%20Matrix%20CNN.png?raw=true" width="400" height ="400" />
</p>

Our Accuracy and Loss graphs vs number of epochs is illustrated below:
<p float="center">
  <img src="https://github.com/Erfan-Mostafiz/CSE499B_EmotionAnalysis/blob/main/Test%20Result/Accuracy%20Graph.png?raw=true" width="500" height ="400" />
  <img src="https://github.com/Erfan-Mostafiz/CSE499B_EmotionAnalysis/blob/main/Test%20Result/Loss%20Graph.png?raw=true" width="500" height ="400" />
</p>
