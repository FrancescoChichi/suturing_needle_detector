import numpy as np
import argparse
import cv2
import imutils
import fitEllipse as elps
import Queue as qe
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())

img_size = [640,480]
threshold_px = 100
dim_roi = [80,80]
upper=[160,180]
lower=[10,20]
threshold_ellipse = 10
threshold_mean=200
stackMean = 5
meanL = [[],[]]
meanR = [[],[]]

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

cv2.namedWindow('image')
# create trackbars for color change
cv2.createTrackbar('tm','image',160,255,nothing)
cv2.createTrackbar('tM','image',255,255,nothing)
cv2.createTrackbar('Hm','image',160,255,nothing)
cv2.createTrackbar('HM','image',255,255,nothing)
cv2.createTrackbar('Sm','image',32,255,nothing)
cv2.createTrackbar('SM','image',91,255,nothing)
cv2.createTrackbar('Vm','image',92,255,nothing)
cv2.createTrackbar('VM','image',255,255,nothing)



while camera != 0:

  (grabbed, frame) = camera.read()

  if not grabbed:
    print("FRAME NOT GRABBED")
    break

  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  # get current positions of four trackbars
  tm = cv2.getTrackbarPos('tm', 'image')
  tM = cv2.getTrackbarPos('tM', 'image')
  hm = cv2.getTrackbarPos('Hm', 'image')
  hM = cv2.getTrackbarPos('HM', 'image')
  sm = cv2.getTrackbarPos('Sm', 'image')
  sM = cv2.getTrackbarPos('SM', 'image')
  vm = cv2.getTrackbarPos('Vm', 'image')
  vM = cv2.getTrackbarPos('VM', 'image')

  lower_range = np.array([hm, sm, vm], dtype=np.uint8)
  upper_range = np.array([hM, sM, vM], dtype=np.uint8)

  thresh = cv2.threshold(frame, tm, tM, cv2.THRESH_BINARY)[1]
  thresh = cv2.erode(thresh, None, iterations=2)
  thresh = cv2.dilate(thresh, None, iterations=2)

  mask = cv2.inRange(hsv, lower_range, upper_range)

  #canny = cv2.Canny(hsv-thresh,100,200)


  cv2.imshow('original frame', frame)
  cv2.imshow('hsv frame', hsv)
  cv2.imshow('threshold', thresh)
  cv2.imshow('m', hsv+thresh)
  #cv2.imshow('c', canny)

  cv2.waitKey(100)


# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
