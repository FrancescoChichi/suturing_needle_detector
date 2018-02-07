import numpy as np
import argparse
import cv2
import imutils
import fitEllipse as elps
import Queue as qe
import fitEllipse as fe
#import cv2.cv as cv
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())
K = np.ndarray(shape=(3,3), dtype=float, order='F')
img_size = [640,480]
threshold_px = 100
dim_roi = [80,80]
upper=[160,180]
lower=[10,20]
threshold_ellipse = 10
threshold_mean=200
stackMean = 20
mean_list = [[],[]]
min_window = [100,100]
max_window = [200,200]
offset = 20
ellipse_size = 40

def nothing(x):
  pass



if args.get("video", None) is None:
  print("no video founded")
  camera = 0
else:
  camera = cv2.VideoCapture(args["video"])
  camera.set(cv2.CAP_PROP_FRAME_WIDTH, img_size[0])
  camera.set(cv2.CAP_PROP_FRAME_HEIGHT, img_size[1])
  camera.set(cv2.CAP_PROP_FPS, 1)

print("opencv version "+cv2.__version__)

cv2.namedWindow('ellipses_thresholds',)
cv2.createTrackbar('cx','ellipses_thresholds',0,250,nothing)
cv2.createTrackbar('cy','ellipses_thresholds',0,250,nothing)
cv2.createTrackbar('h','ellipses_thresholds',0,250,nothing)
cv2.createTrackbar('w','ellipses_thresholds',0,250,nothing)
cv2.createTrackbar('d','ellipses_thresholds',0,250,nothing)





while camera != 0:

  (grabbed, current_frame) = camera.read()

  center = tuple([cv2.getTrackbarPos('cx', 'ellipses_thresholds'),cv2.getTrackbarPos('cy', 'ellipses_thresholds')])
  axes   = tuple([cv2.getTrackbarPos('h', 'ellipses_thresholds'),cv2.getTrackbarPos('w', 'ellipses_thresholds')])
  d = cv2.getTrackbarPos('d', 'ellipses_thresholds')

  if not grabbed:
    print("FRAME NOT GRABBED")
    break

  ellipse = tuple([center,axes,d])
  cv2.ellipse(current_frame,ellipse,(0,0,255),2)

  cv2.imshow('a', current_frame)
  cv2.waitKey(0)


  
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

