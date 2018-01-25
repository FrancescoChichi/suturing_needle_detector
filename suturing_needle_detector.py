import numpy as np
import argparse
import cv2
import imutils

# from keras.models import load_model
# from matplotlib import pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())
K = np.ndarray(shape=(3,3), dtype=float, order='F')
img_size = [640,480]
threshold_px = 50

def nothing(x):
  pass

def getContour(img_gray):
  gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
 
  # threshold the image, then perform a series of erosions +
  # dilations to remove any small regions of noise
  thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
  thresh = cv2.erode(thresh, None, iterations=2)
  thresh = cv2.dilate(thresh, None, iterations=2)
   
  # find contours in thresholded image, then grab the largest
  # one
  cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)

  cnts = cnts[0] if imutils.is_cv2() else cnts[1]
  return max(cnts, key=cv2.contourArea)


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
  camera.set(cv2.CAP_PROP_FRAME_WIDTH, img_size[0])
  camera.set(cv2.CAP_PROP_FRAME_HEIGHT, img_size[1])
  camera.set(cv2.CAP_PROP_FPS, 1)

print("opencv version "+cv2.__version__)

(grabbed, current_frame) = camera.read()

previous_frame = current_frame

first = True

m=img_size[0]/2

first_c = getContour(cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY))
top = tuple(first_c[first_c[:, :, 1].argmin()][0])
bot = tuple(first_c[first_c[:, :, 1].argmax()][0])

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


  current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
  previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)


  frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)

  c = getContour(current_frame_gray)
 
  right_ef = [10000,1000]
  left_ef = [-1,-1]

  extTop = tuple(c[c[:, :, 1].argmin()][0])
  extBot = tuple(c[c[:, :, 1].argmax()][0])

  left = c[:c[:, :, 1].argmax()+1,0,:]
  right = c[c[:, :, 1].argmax():,0,:]


  for p in range(len(left)): 
    if not((left[p][1]>=top[1]-threshold_px  and left[p][1]<=top[1]+threshold_px) or (left[p][1]>=bot[1]-threshold_px)):
      if(left[p][0] >= left_ef[0]):
        left_ef=left[p]

  for p in range(len(right)): 
    if not(((right[p][1]>=top[1]-threshold_px  and right[p][1]<=top[1]+threshold_px) or (right[p][1]>=bot[1]-threshold_px))):
      if(right[p][0] <= right_ef[0]):
        right_ef=right[p]

  left_ef = tuple([left_ef[0],left_ef[1]])
  right_ef = tuple([right_ef[0],right_ef[1]])

  #extLeft = tuple(c[c[:, :, 0].argmax()][0])

  #cv2.drawContours(current_frame, [c], -1, (0, 255, 0), 2)

  cv2.circle(current_frame, right_ef, 8, (0, 0, 255), -1)
  cv2.circle(current_frame, left_ef, 8, (0, 255, 255), -1)



    
  roi = current_frame[left_ef[1]:left_ef[1]+(right_ef[1]-left_ef[1]),   left_ef[0]:left_ef[0]+(right_ef[0]-left_ef[0])]

  edges = cv2.Canny(frame_diff, 100, 200)
  #cv2.imshow('mask', current_frame_gray)
  #cv2.imshow('frame diff', frame_diff)
  #cv2.imshow('edges', edges)
  cv2.imshow('original frame', current_frame)
  cv2.imshow('ROI', roi)

  cv2.waitKey(1)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  #previous_frame_gray = current_frame_gray.copy()
  previous_frame = current_frame.copy()

  grabbed, current_frame = camera.read()
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
