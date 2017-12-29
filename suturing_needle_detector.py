import numpy as np
import argparse
import cv2
import time

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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
  camera = cv2.VideoCapture(0)
  time.sleep(0.25)

# otherwise, we are reading from a video file
else:
  camera = cv2.VideoCapture(args["video"])

# initialize the first frame in the video stream
boundaries = [
	([17, 15, 100], [50, 56, 200]),
	([86, 31, 4], [220, 88, 50]),
	([25, 146, 190], [62, 174, 250]),
	([103, 86, 65], [145, 133, 128])
]

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
fgbg = cv2.createBackgroundSubtractorMOG2()
# loop over the boundaries
# loop over the frames of the video
while True:
  # grab the current frame and iniap.add_argument("-v", "--video", help="path to the video file")tialize the occupied/unoccupied
  # text
  (grabbed, frame) = camera.read()

  fgmask = fgbg.apply(frame)
  dfgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
  #firstFrame = frame

  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  # get current positions of four trackbars
  hm = cv2.getTrackbarPos('Hm', 'image')
  hM = cv2.getTrackbarPos('HM', 'image')
  sm = cv2.getTrackbarPos('Sm', 'image')
  sM = cv2.getTrackbarPos('SM', 'image')
  vm = cv2.getTrackbarPos('Vm', 'image')
  vM = cv2.getTrackbarPos('VM', 'image')
  # if the frame could not be grabbed, then we have reached the end
  # of the video
  if not grabbed:
    break

  lower_range = np.array([hm, sm, vm], dtype=np.uint8)
  upper_range = np.array([hM, sM, vM], dtype=np.uint8)

  mask = cv2.inRange(hsv, lower_range, upper_range)

  #roi = frame[frame.size - 20: frame.size + 20, frame.size - 20: frame.size + 20]
  #cv2.imshow('ROI', roi)
  cv2.imshow('bs', fgmask)
  cv2.imshow('mask', mask)
  cv2.imshow("frame", frame)
  #cv2.imshow("images", np.hstack([frame, output]))
  #cv2.waitKey(0)

  key = cv2.waitKey(1) & 0xFF

  # if the `q` key is pressed, break from the lop
  if key == ord("q"):
    break


  '''for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(frame, lower, upper)
    output = cv2.bitwise_and(frame, frame, mask=mask)

    # show the images
    cv2.imshow("images", np.hstack([frame, output]))
    cv2.waitKey(0)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
      break
  '''
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()