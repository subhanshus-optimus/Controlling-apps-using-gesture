import cv2
import time

import numpy as np
from nextArrow import nextpage
from previuosKey import previousPage
from keras.models import load_model



cap = cv2.VideoCapture(0)
model = load_model('./modelFiles/3.h5')


img_counter = 0


x_start=160
y_start=60
x_end=430
y_end=350

while True:
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    kernel = np.ones((3,3),np.uint8)
    
    #define region of interest
    roi=frame[y_start:y_end, x_start:x_end]
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
#        frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame,(x_start,y_start),(x_end,y_end),(0,255,0),0)

      
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    
     
# define range of skin color in HSV
    lower_skin = np.array([0,20,70], dtype=np.uint8)
    upper_skin = np.array([20,255,255], dtype=np.uint8)
    
 #extract skin colur imagw  
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
   
    
#extrapolate the hand to fill dark spots within
    mask = cv2.dilate(mask,kernel,iterations = 4)
    
#blur the image
    mask = cv2.GaussianBlur(mask,(5,5),100) 
    
    cv2.imshow('mask',mask)
    cv2.imshow('frame',frame)
    if not ret:
        break
    k = cv2.waitKey(10)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    else:

        
        mask = cv2.resize(mask, (56, 56)) 
        x_train =np.asarray(mask)
        pre = model.predict(x_train.reshape(1,x_train.shape[0],x_train.shape[1],1))

        returnedValue=np.argmax(pre)
        print(returnedValue)        
        time.sleep(0.30)
        
        if (returnedValue==0):

            nextpage()

        elif(returnedValue==1):

            previousPage()

        else:

            k=1


cap.release()

cv2.destroyAllWindows()







