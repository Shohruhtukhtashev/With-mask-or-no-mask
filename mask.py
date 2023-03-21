# required libraries
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import cvlib as cv
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf

model = load_model('mask.model') # model load
turi = ['Mask','No_mask'] # label

video = cv2.VideoCapture("Untitled video - Made with Clipchamp.mp4") # video was used for the test.

while video.isOpened():
    boolean, kadr = video.read() # read the video
    face,confindence = cv.detect_face(kadr)
    
    for index,yuz in enumerate(face):
        (startX,startY,endX,endY) = yuz[0],yuz[1],yuz[2],yuz[3] # determine the coordinates
        
        yuz_np = np.copy(kadr[startY:endY,startX:endX]) # copy
        
        if yuz_np.shape[0] < 10 or yuz_np.shape[1] < 10: # if the face size is less than 10, continue
            continue                                     # If it doesn't work, it continues with the next code
            
        yuz_np = cv2.resize(yuz_np,(224,224))       # resizing face_np to 224,224
        yuz_np = yuz_np.astype("float") / 255.0     # normalization
        yuz_np = img_to_array(yuz_np)
        yuz_np = np.expand_dims(yuz_np,axis=0)
        
        bashorat = model.predict(yuz_np)[0] #predict the model
        index = np.argmax(bashorat)         # the index of the maximum value
        label = turi[index]                 #we get the label by index from the list
        
        # if it has a mask, it draws a green color.
        if label == 'Mask':
            color = (0,255,0)
        # otherwise it draws red
        else:
            color = (0,0,255)
            
        label = f"{label} {np.around(bashorat[index]*100,2)}"
        
        if startY-10 > 10:
            Y=startY-10
        else:
            Y=startY+10
        
        cv2.rectangle(kadr,(startX,startY),(endX,endY),color,2)                 # draw rectangle
        cv2.putText(kadr,label,(startX,Y),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2) # writing the label
        
    cv2.imshow("Maska",kadr) # view the resulting image
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # stop the process
        break
video.release()
cv2.destroyAllWindows()
