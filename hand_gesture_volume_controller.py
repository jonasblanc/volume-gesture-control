#!/usr/bin/env python
# coding: utf-8

import cv2
import math
import os
import time
import itertools
import subprocess
from subprocess import call
import re
import copy
import numpy as np

import tensorflow as tf
import mediapipe

import tkinter as tk
from tkinter import *
from tkinter.ttk import Label, Progressbar, Style


drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

SHOW_CAMERA = False
VOLUME_STEP_PER_TIME_STEP_SEC = 5
TIME_STEP_SEC = 0.5


def get_speaker_output_volume():
    """
    Get the current speaker output volume from 0 to 100.

    Note that the speakers can have a non-zero volume but be muted, in which
    case we return 0 for simplicity.

    Note: Only runs on macOS.
    """
    cmd = "osascript -e 'get volume settings'"
    process = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
    output = process.stdout.strip().decode('ascii')

    pattern = re.compile(r"output volume:(\d+), input volume:(\d+), "
                         r"alert volume:(\d+), output muted:(true|false)")
    volume, _, _, muted = pattern.match(output).groups()

    volume = int(volume)
    muted = (muted == 'true')

    return 0 if muted else volume

def set_speaker_output_volume(volume):
    """
    Set the volume to a value from 0 to 100
    
    Note: Only runs on macOS. 
    """
    volume = max(0, min(100, volume))
    call([f"osascript -e 'set volume output volume {volume}'"], shell=True)

    
class KeyPointClassifier(object):
    """
    Classify hand keys points into 8 gestures
    
    Note: the classification model and class has been taken and refactored from https://github.com/kinivi/tello-gesture-control
    """
    def __init__(
        self,
        model_path="keypoint_classifier.tflite",
        
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        frame,
        hand_landmarks,
    ):
        # Landmark calculation
        landmark_list = self._calc_landmark_list(frame, hand_landmarks)

        # Conversion to relative coordinates / normalized coordinates
        pre_processed_landmark_list = self._pre_process_landmark(landmark_list)

        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([pre_processed_landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index
    
    def _pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list
    
    def _calc_landmark_list(self, image, landmarks):
            image_width, image_height = image.shape[1], image.shape[0]

            landmark_point = []

            # Keypoint
            for _, landmark in enumerate(landmarks.landmark):
                landmark_x = min(int(landmark.x * image_width), image_width - 1)
                landmark_y = min(int(landmark.y * image_height), image_height - 1)
                # landmark_z = landmark.z

                landmark_point.append([landmark_x, landmark_y])

            return landmark_point


def gesture_handler(gesture_idx):
    """
    Return the new volume based one the input gesture index
    """
    target_volume = get_speaker_output_volume()

    action_taken = True
    # UP
    if(gesture_idx == 2):
        target_volume += VOLUME_STEP_PER_TIME_STEP_SEC

    # DOWN
    elif(gesture_idx == 4):
        target_volume -= VOLUME_STEP_PER_TIME_STEP_SEC

    # CLOSE FIST
    elif(gesture_idx == 7):
        target_volume = 0
        
    # OK
    elif(gesture_idx == 3):
        target_volume = 40
        
    else:
        action_taken = False
        
    if action_taken:
        target_volume = max(0, min(100, target_volume))
        
    return target_volume, action_taken

root = tk.Tk()
root.title('Volume control')
root.geometry('500x300')
root.resizable(True, True)

pb = Progressbar(
    root,
    orient='horizontal',
    mode='determinate',
    length=300,
)

pb['value'] = get_speaker_output_volume()


def camera_volume_control():
    volume = get_speaker_output_volume()
    
    gesture_last_check_time = time.time()
    display_last_modif_time = time.time()
    target_time = time.time()
    

    cap = cv2.VideoCapture(0)
    cap.set(3,640) # adjust width
    cap.set(4,480) # adjust height

    key_point_classifier = KeyPointClassifier()

    with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1
                          ) as hands:
                
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)  #Horizontal Flip

            # Extract hand key points
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks != None:

                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):
                    # Display the hand key points 
                    if SHOW_CAMERA:
                        drawingModule.draw_landmarks(frame, hand_landmarks, handsModule.HAND_CONNECTIONS)

                    # Allow a given number of action per second
                    curr_time = time.time()
                    if curr_time - gesture_last_check_time >= TIME_STEP_SEC:
                        gesture_last_check_time = curr_time

                        #  Classify gesture based on the hand key points
                        gesture_idx = key_point_classifier(frame, hand_landmarks)
                        volume, action_taken = gesture_handler(gesture_idx)
                        
                        # If the gesture require an action
                        if action_taken:
                            # Update computer volume
                            set_speaker_output_volume(volume)
                            
                            # Update the progressbar
                            pb['value'] = volume
                            root.update()
            
            # Display the webcam with the hand key points in a new window 
            if SHOW_CAMERA:
                cv2.imshow("Webcam", frame) 
                if cv2.waitKey(1) & 0xFF==ord('q'): # quit when 'q' is pressed
                    cap.release()
                    break

        if SHOW_CAMERA:
            cv2.destroyAllWindows() 
            cv2.waitKey(1)
                
# Interface configuration
root.columnconfigure(0, weight=1)
   
value_label = Label(root, text="👆🏼: volume up \n\n👇🏼: volume down \n\n👌🏼: default volume")
value_label.grid(column=0, row=2, padx=10, pady=10)

pb.grid(column=0, row=1, padx=10, pady=10)

btn = Button(root, text='Start', bd='5',
             command= camera_volume_control)
btn.grid(column=0, row=0, padx=10, pady=10)

root.mainloop()