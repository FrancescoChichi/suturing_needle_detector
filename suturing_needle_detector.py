import numpy as np
import argparse
import cv2
from keras.models import load_model
from matplotlib import pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())

if args.get("video", None) is None:
  print("no video founded")
  camera = 0
else:
  camera = cv2.VideoCapture(args["video"])
  camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  camera.set(cv2.CAP_PROP_FPS, 1)

  print("opencv version "+cv2.__version__)


(grabbed, current_frame) = camera.read()
#cv2.resize(current_frame, current_frame, cv2.Size(640, 360), 0, 0, cv2.INTER_CUBIC)

previous_frame = current_frame
fgbg = cv2.createBackgroundSubtractorMOG2(20,250, False)

with open("/home/francesco/Desktop/medical_robotics/dataset/Suturing/kinematics/AllGestures/Suturing_B001.txt") as f:
  content = f.readlines()
model1 = load_model("/home/francesco/Downloads/rete_medical_2.h5")
myTrainData = []
for i in range(len(content)):
  content[i] = content[i].replace("\n", "")
  content[i] = content[i].split(" ")

  while (content[i].count("") != 0):
    content[i].remove("")
  content[i] = content[i][0:3]
  for k in range(len(content[i])):
    content[i][k] = float(content[i][k])
  myTrainData.append(np.asarray(content[i]))
myTrainData = np.array(myTrainData)
i = 0

while camera != 0:

  if not grabbed:
    break

  predictList = []
  predictList.append(myTrainData[i])
  predictList = np.array(predictList)
  a = model1.predict(predictList)
  cv2.circle(current_frame, (a[0][0],a[0][1]), 3, (0,255,0))
  #print(a)

  i = i+1

  current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
  previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

  fgmask = fgbg.apply(current_frame)
  cv2.imshow("cazzoo",fgmask)

  '''
  aperture = 3

  #dst = current_frame_gray
  dst = cv2.Laplacian(current_frame, cv2.CV_64F, 3)

  gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
  thresholded = current_frame

  cv2.imshow('gray ', gray)
  '''

  '''
  sobelx = cv2.Sobel(current_frame_gray, cv2.CV_32F, 1, 0, ksize=5)
  sobely = cv2.Sobel(current_frame_gray, cv2.CV_32F, 0, 1, ksize=5)
  mag, angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
  cv2.imshow('mag ', mag)
  cv2.imshow('angle ', angle)
  cv2.imshow('sobel x ', sobelx)
  cv2.imshow('sobel y ', sobely)
  '''


  frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)

  #cv2.imshow('frame diff ', frame_diff)
  cv2.imshow('original frame ', current_frame)

  cv2.waitKey(0)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  previous_frame = current_frame.copy()

  grabbed, current_frame = camera.read()
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
