# Hand gesture controller for computer sound volume

I usually project the films I watch using a projector from my computer. Once I'm comfortable under the duvet, there's nothing more annoying than having to get up and adjust the sound level. As an engineer I felt compelled to provide a solution to this vital problem. So I offer you to adjust the sound of your computer by hand gestures using your webcam.  

https://user-images.githubusercontent.com/44334351/161393693-fdef9110-170b-404b-a418-0ef7f88f50a0.mov

For this project I used [mediaPipe](https://google.github.io/mediapipe/) which offers cross-platform ML solutions. I used this powerful library to extract the hand position. Then to recognize the different control gestures, I used a machine learning model on [tensorflow](https://www.tensorflow.org) developed and trained for [tello-gesture-control](https://github.com/kinivi/tello-gesture-control). If you are interested in this part, please read their git/blog everything is very well explained. Many thanks to them for this model, to be honest I had no desire to recreate a labelled dataset. Finally, I created a minimalist interface to control the whole thing.

Note: everything was developed for a Mac, if someone wants to adapt it for window feel free to contribute. 
