import numpy as np
import argparse
import cv2

def nothing(x):
  pass

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())

if args.get("video", None) is None:
  print("no video founded")
  camera = 0;
else:
  camera = cv2.VideoCapture(args["video"])

(grabbed, current_frame) = camera.read()
previous_frame = current_frame

while camera != 0:

  if not grabbed:
    break

  current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
  previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

  frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)

  cv2.imshow('frame diff ', frame_diff)
  cv2.imshow('original frame ', current_frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  previous_frame = current_frame.copy()


  grabbed, current_frame = camera.read()

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()