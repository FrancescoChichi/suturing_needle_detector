import numpy as np
import argparse
import cv2

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
fgbg = cv2.createBackgroundSubtractorMOG2(20, 250, False)
with open("/home/francesco/Desktop/2d") as f2d:
  content2d = f2d.readlines()

with open("/home/francesco/Desktop/3d") as f3d:
  content3d = f3d.readlines()

points2d = []
for i in range(len(content2d)):
  content2d[i] = content2d[i].replace("\n", "")
  content2d[i] = content2d[i].split(',')
  while content2d[i].count("") != 0:
    content2d[i].remove("")

  for k in range(len(content2d[i])):
    content2d[i][k] = float(content2d[i][k])

  pt = np.array([content2d[i][0], content2d[i][1]]).astype('float32')
  points2d.append(pt)
imgpoints = np.array(points2d)

points3d = []
for i in range(len(content3d)):
  content3d[i] = content3d[i].replace("\n", "")
  content3d[i] = content3d[i].split(',')
  while content3d[i].count("") != 0:
    content3d[i].remove("")

  for k in range(len(content3d[i])):
    content3d[i][k] = float(content3d[i][k])

  pt = np.array([content2d[i][0], content2d[i][1], content3d[i][2]]).astype('float32')
  points3d.append(pt)
objpoints = np.array(points3d)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objpoints], [imgpoints], (640,480) ,None,None,flags=cv2.CALIB_USE_INTRINSIC_GUESS)
print(mtx)
# p
# rint(dist)