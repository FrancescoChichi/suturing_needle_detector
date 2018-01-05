'''
Created on 04 gen 2018

@author: emanuele
'''
import keras
from keras.models import load_model
import json
import cv2
from keras.models import model_from_json
import numpy as np

with open("Suturing_B001.txt", encoding="utf8") as f:
  content = f.readlines()
model1 = load_model("rete_medical_2.h5")
myTrainData = []
for i in range(len(content)):
  content[i] = content[i].replace("\n","")
  content[i] = content[i].split(" ")

  while(content[i].count("") != 0):
    content[i].remove("")
  content[i] = content[i][0:3]
  for k in range(len(content[i])):
    content[i][k] = float(content[i][k])
  myTrainData.append(np.asarray(content[i]))
myTrainData = np.array(myTrainData)

#print(content[0])



for i in range(1000):
  predictList = []
  predictList.append(myTrainData[i])
  predictList = np.array(predictList)
  a = model1.predict(predictList)
  print(a)