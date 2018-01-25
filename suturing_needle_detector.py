import numpy as np
import argparse
import cv2
# from keras.models import load_model
# from matplotlib import pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())
K = np.ndarray(shape=(3,3), dtype=float, order='F')


def nothing(x):
  pass

cv2.namedWindow('image')
# create trackbars for color change
cv2.createTrackbar('Hm','image',0,255,nothing)
cv2.createTrackbar('HM','image',0,255,nothing)
cv2.createTrackbar('Sm','image',0,255,nothing)
cv2.createTrackbar('SM','image',0,255,nothing)
cv2.createTrackbar('Vm','image',0,255,nothing)
cv2.createTrackbar('VM','image',0,255,nothing)

cv2.setTrackbarPos('Hm', 'image',0)
cv2.setTrackbarPos('HM', 'image',51)
cv2.setTrackbarPos('Sm', 'image',11)
cv2.setTrackbarPos('SM', 'image',198)
cv2.setTrackbarPos('Vm', 'image',65)
cv2.setTrackbarPos('VM', 'image',86)

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

previous_frame = current_frame

first = True

while camera != 0:

  if not grabbed:
    print("FRAME NOT GRABBED")
    break

  hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

  #get current positions of four trackbars
  hm = cv2.getTrackbarPos('Hm', 'image')
  hM = cv2.getTrackbarPos('HM', 'image')
  sm = cv2.getTrackbarPos('Sm', 'image')
  sM = cv2.getTrackbarPos('SM', 'image')
  vm = cv2.getTrackbarPos('Vm', 'image')
  vM = cv2.getTrackbarPos('VM', 'image')

  lower_range = np.array([hm, sm, vm], dtype=np.uint8)
  upper_range = np.array([hM, sM, vM], dtype=np.uint8)

  current_frame_gray = cv2.inRange(hsv, lower_range, upper_range)

  # #img moments
  # ret, thresh = cv2.threshold(current_frame_gray, 127, 255, 0)
  # contours = cv2.findContours(thresh, 1, 2)
  # cnt = contours[0]
  # M = cv2.moments(cnt)
  # print M
  #
  # cx = int(M['m10'] / M['m00'])
  # cy = int(M['m01'] / M['m00'])
  # area = cv2.contourArea(cnt)
  # perimeter = cv2.arcLength(cnt, True)
  # epsilon = 0.1 * cv2.arcLength(cnt, True)
  # approx = cv2.approxPolyDP(cnt, epsilon, True)


  cv2.imshow('mask', current_frame_gray)

  '''
  if(first):
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    first = False
  '''
  current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
  previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)


  frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)

  edges = cv2.Canny(frame_diff, 100, 200)
  cv2.imshow('frame diff', frame_diff)
  cv2.imshow('edges', edges)
  cv2.imshow('original frame', current_frame)

  cv2.waitKey(0)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  #previous_frame_gray = current_frame_gray.copy()
  previous_frame = current_frame.copy()

  grabbed, current_frame = camera.read()
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
