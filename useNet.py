'''
Created on 04 gen 2018

@author: emanuele
'''
import keras
from keras.models import load_model
import json
from keras.models import model_from_json
import numpy as np
import cv2
import math
from cmath import sqrt
from numpy.distutils.fcompiler import none
from skimage import data, color
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte


def nothing(x):
  pass

def classify(image, model):
    i = 0
    k = 0
    slidingPass = 100
    pointList = []
    a = []
    subImage = []
    #print(len(image))
    #print(len(image[i]))
    while(i < len(image )):
        subImage = []
        k = 0
        while(k < len(image[i])):
            a = []
            if((i +slidingPass) < len(image) and (k +slidingPass)< len(image[i])):
                subImage = image[i:i+slidingPass,k:k+slidingPass]
                a.append(subImage)
                a = np.array(a)
                res = int(model.predict(a))
                if(res == 1):
                    point = [int(k +(slidingPass/2)), int(i+(slidingPass/2))]
                    pointList.append(point)
                #print(k)
                #print(res)
            k = k + slidingPass
        
        i = i + slidingPass
    return pointList 
classModel = load_model("medical_conv_6_gray.h5")
##DEFINITIONS
UseRoi = 1
UseCircles = 0
UseEdges = 1
slidingPass = 12

# cv2.namedWindow('image')
# # create trackbars for color change
# cv2.createTrackbar('Hm','image',0,255,nothing)
# cv2.createTrackbar('HM','image',0,255,nothing)
# cv2.createTrackbar('Sm','image',0,255,nothing)
# cv2.createTrackbar('SM','image',0,255,nothing)
# cv2.createTrackbar('Vm','image',0,255,nothing)
# cv2.createTrackbar('VM','image',0,255,nothing)



with open("Suturing_B001.txt", encoding="utf8") as f:
    content = f.readlines()
model1 = load_model("rete_medical_13_L.h5")    

model2 = load_model("rete_medical_1_R.h5") 
myTrainData = []
myTrainData2 = []
for i in range(len(content)):
    content[i] = content[i].replace("\n","")
    content[i] = content[i].split(" ")
    
    while(content[i].count("") != 0):
        content[i].remove("")
    #content[i] = content[i][0:3] + content[i][12:15]
    content2 = content[i][19:22]
    content[i] = content[i][0:19]
    for k in range(len(content[i])):
        content[i][k] = float(content[i][k])
    for k in range(len(content2)):
        content2[k] = float(content2[k])
    content[i][0] = content[i][0]/content[i][2]
    content[i][1] = content[i][1]/content[i][2] 
    
    content2[0] = content2[0]/content2[2]
    content2[1] = content2[1]/content2[2]
    
    content[i] = content[i][0:2]
    content2 = content2[0:2]
    
    myTrainData2.append(np.asarray(content2))
    myTrainData.append(np.asarray(content[i]))
    
myTrainData = np.array(myTrainData)
myTrainData2 = np.array(myTrainData2)
#print(content[0])

cap = cv2.VideoCapture("Suturing_B001_capture1.avi")
i = 0
predictListR = []


bcksub = cv2.createBackgroundSubtractorKNN()
while(True):
    
    
    
    ret, frame = cap.read()
    
    
    
    i = i +1
    predictList = []
    predictListR = []
    predictList.append(myTrainData[i])
    predictList = np.array(predictList)
    
    predictListR.append(myTrainData2[i])
    predictListR = np.array(predictListR)
    a = model1.predict(predictList)
    a2 = model2.predict(predictListR)
    
     # Our operations on the frame come here
    gray = frame
    #gray = cv2.threshold(frame,127,255,cv2.THRESH_BINARY)
    centerX = (a[0][0] + a2[0][0])/2
    centerY = (a[0][1] + a2[0][1])/2
    radius = ((a[0][0] - a2[0][0]) * (a[0][0] - a2[0][0])) + ((a[0][1] - a2[0][1]) * (a[0][1] - a2[0][1]))
    
    radius = math.sqrt(radius) 
    
    radius = int(radius)
    centerX = int(centerX)
    centerY = int(centerY)
    if(UseRoi == 1 and UseCircles == 0):
        if(radius < 220):
            lowerX = 0
            lowerY = 0
            if(centerX - radius > 0):
                lowerX = centerX - radius
            
            if(centerY - radius > 0):
                lowerY = centerY - radius
            croppedImage = gray[lowerY:centerY+radius, lowerX:centerX+radius]
            
            saveCropped = croppedImage
            saveCropped = np.asarray(saveCropped)
            saveCropped = cv2.cvtColor(saveCropped, cv2.COLOR_BGR2GRAY)
            pointList = classify(saveCropped, classModel)
            print(pointList)
            
            
            if(UseEdges == 1):
                edges = cv2.Canny(croppedImage,30,200)
                
                
                #print(cx)
                for i in range(len(pointList)):
                    cv2.circle(croppedImage,(pointList[i][0],pointList[i][1]), 10, (255,255,0))
                cv2.imshow('frame',croppedImage)
                
            else:
                cv2.imshow('frame',croppedImage)
            
        else:
            radius = 130
            
            centerX_1 = int(a[0][0])
            centerY_1 = int(a[0][1])
            centerX_2 = int(a2[0][0])
            centerY_2 = int(a2[0][1])
            
            lowerY1 = 0
            lowerY2 = 0
            
            lowerX1 = 0
            lowerX2 = 0
            
            if(centerX_1 - radius > 0):
                lowerX1 = centerX_1 - radius
            if(centerY_1 - radius > 0):
                lowerY1 = centerY_1 - radius
            if(centerX_2 - radius > 0):
                lowerX2 = centerX_2 - radius
            if(centerY_2 - radius > 0):
                lowerY2 = centerY_2 - radius
            
            croppedImage1 = gray[lowerY1:centerY_1+radius, lowerX1:centerX_1+radius]
            
            croppedImage2 = gray[lowerY2:centerY_2+radius, lowerX2:centerX_2+radius]
            
            
            if(UseEdges == 1):
                edges1 = cv2.Canny(croppedImage1,100,200)
                cv2.imshow('frame_left',edges1)
                edges2 = cv2.Canny(croppedImage2,100,200)
                cv2.imshow('frame_right',edges2)
            else:
                cv2.imshow('frame_Left',croppedImage1)
                cv2.imshow('frame_Right',croppedImage2)            
        
    if(UseRoi == 0 and UseCircles == 1):
        if(radius < 220):
            cv2.circle(gray,(centerX,centerY), radius, (0,255,0),thickness=1, lineType=8, shift=0)
        else:
            cv2.circle(gray,(a[0][0],a[0][1]), 130, (0,255,0),thickness=1, lineType=8, shift=0)
            cv2.circle(gray,(a2[0][0],a2[0][1]), 130, (0,255,0),thickness=1, lineType=8, shift=0)
        # Display the resulting frame
        cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
