import numpy as np
import argparse
import cv2
import imutils
import fitEllipse as elps
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

def getContour(img_gray,a=100):
  gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
 
  # threshold the image, then perform a series of erosions +
  # dilations to remove any small regions of noise
  thresh = cv2.threshold(gray, a, 255, cv2.THRESH_BINARY)[1]
  thresh = cv2.erode(thresh, None, iterations=2)
  thresh = cv2.dilate(thresh, None, iterations=2)
   
  # find contours in thresholded image, then grab the largest
  # one
  cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)

  cnts = cnts[0] if imutils.is_cv2() else cnts[1]
  return max(cnts, key=cv2.contourArea)


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

  current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
  previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

  c = getContour(current_frame_gray)
 
  right_ef = [10000,1000]
  left_ef = [-1,-1]

  left = c[:c[:, :, 1].argmax()+1,0,:]
  lx = np.array([])
  ly = np.array([])
  right = c[c[:, :, 1].argmax():,0,:]
  rx = np.array([])
  ry = np.array([])

  for p in range(len(left)): 
    if not((left[p][1]>=top[1]-threshold_px  and left[p][1]<=top[1]+threshold_px) or (left[p][1]>=bot[1]-threshold_px)):
      lx=np.append(lx,left[p][0])
      ly=np.append(ly,left[p][1])
      if(left[p][0] >= left_ef[0]):
        left_ef=left[p]

  for p in range(len(right)): 
    if not(((right[p][1]>=top[1]-threshold_px  and right[p][1]<=top[1]+threshold_px) or (right[p][1]>=bot[1]-threshold_px))):
      rx=np.append(rx,right[p][0])
      ry=np.append(ry,right[p][1])
      if(right[p][0] <= right_ef[0]):
        right_ef=right[p]

  left_ef = tuple([left_ef[0],left_ef[1]])
  right_ef = tuple([right_ef[0],right_ef[1]])

  ellipse=elps.fitEllipse(rx,ry)
  center = elps.ellipse_center(ellipse)
  phi = elps.ellipse_angle_of_rotation2(ellipse)
  axes = elps.ellipse_axis_length(ellipse)
  center = tuple([int(center[0]),int(center[1])])
  print ellipse
  #cv2.drawContours(current_frame, [ellipse], -1, (255, 255, 0), 2)

  #extLeft = tuple(c[c[:, :, 0].argmax()][0])

  #cv2.drawContours(current_frame, [c], -1, (0, 255, 0), 2)




  dim_roi = [70,70]
  roi_r = current_frame[right_ef[1]-dim_roi[1]:right_ef[1]+dim_roi[1], right_ef[0]-dim_roi[0]:right_ef[0]+dim_roi[0]]
  roi_l = current_frame[left_ef[1]-dim_roi[1]:left_ef[1]+dim_roi[1], left_ef[0]-dim_roi[0]:left_ef[0]+dim_roi[0]]
  edges_r = cv2.Canny(roi_r, 100, 200)
  edges_l = cv2.Canny(roi_l, 100, 200)


  #ellipse = cv2.fitEllipse(r)
  #cv2.ellipse(current_frame,ellipse,(0,255,0),2)

  roi_rg = cv2.cvtColor(roi_r, cv2.COLOR_BGR2GRAY)
  r = getContour(roi_rg,150)
  cv2.drawContours(roi_r, [r], -1, (0, 255, 0), 2)
 


  roi_lg = cv2.cvtColor(roi_l, cv2.COLOR_BGR2GRAY)
  l = getContour(roi_lg,150)
  cv2.drawContours(roi_l, [l], -1, (0, 255, 255), 2)
  #ellipse = cv2.fitEllipse(l)
  #cv2.ellipse(roi_l,ellipse,(0,255,255),2)

  #frame_diff = cv2.absdiff(black, previous_frame_gray)

  #cv2.imshow('ROI', roi_rg)

  #cv2.imshow('frame diff', frame_diff)
  #cv2.imshow('edges_l', edges_l)
  #cv2.imshow('edges_r', edges_r)


  '''
  l = getContour(cv2.cvtColor(roi_l, cv2.COLOR_BGR2GRAY))
  cv2.drawContours(roi_l, [l], -1, (0, 255, 0), 2)
  

  r = getContour(cv2.cvtColor(roi_r, cv2.COLOR_BGR2GRAY))
  cv2.drawContours(roi_r, [l], -1, (0, 255, 255), 2)
  '''

  cv2.circle(current_frame, right_ef, 8, (0, 0, 255), -1)
  cv2.circle(current_frame, left_ef, 8, (0, 255, 255), -1)
  #cv2.circle(current_frame, center, 8, (255, 255, 0), -1)

  cv2.imshow('original frame', current_frame)
  #cv2.imshow('left', roi_l)
  #cv2.imshow('right', roi_r)

  cv2.waitKey(0)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  #previous_frame_gray = current_frame_gray.copy()
  previous_frame = current_frame.copy()

  grabbed, current_frame = camera.read()
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
