# Real-Time Emotion Recognition from Frontal Face Videos using Deep Learning Techniques

This repository contains all the source code of my final semester thesis research project of real-time emotion detection. </br>
Used Deep Learning Techniques and our CNN based model achieved a `Test Accuracy` of `75.4%` on the FER2013 dataset

<p align = "center">
  The following Image illustrates the Real Time Emotion Detection Process of our system 
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

