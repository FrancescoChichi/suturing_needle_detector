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
threshold_px = 100
dim_roi = [100,100]

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
  cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE)

  a,b,c = cv2.findContours(thresh.copy(), cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE)

  print c.shape
  cnts = cnts[0] if imutils.is_cv2() else cnts[1]
  return max(cnts, key=cv2.contourArea)

def findCentralPoint(direction, v, ef, top, bot, x=None, y=None): #0 right, 1 left
  if not(x and y):
    skip = True
  else:
    skip = False

  if(direction):
    for p in range(len(v)): 
      if not((v[p][1]>=top[1]-threshold_px  and v[p][1]<=top[1]+threshold_px) or (v[p][1]>=bot[1]-threshold_px)):
        if not(skip):
          x=np.append(x,v[p][0])
          y=np.append(y,v[p][1])
        if(v[p][0] >= ef[0]):
          ef=v[p]
  else :
    for p in range(len(v)): 
      if not(((v[p][1]>=top[1]-threshold_px  and v[p][1]<=top[1]+threshold_px) or (v[p][1]>=bot[1]-threshold_px))):
        if not(skip):  
          x=np.append(x,v[p][0])
          y=np.append(y,v[p][1])
        if(v[p][0] <= ef[0]):
          ef=v[p]
  return tuple([ef[0],ef[1]]), x, y

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

  right_ef = [10000,10000]
  left_ef = [-1,-1]

  left = c[:c[:, :, 1].argmax()+1,0,:]
  lx = np.array([])
  ly = np.array([])
  right = c[c[:, :, 1].argmax():,0,:]
  rx = np.array([])
  ry = np.array([])

  left_ef, lx, ly = findCentralPoint(1, left, left_ef, top, bot, lx, ly) #0 right, 1 left
  right_ef, rx, ry = findCentralPoint(0, right, right_ef, top, bot, rx, ry) #0 right, 1 left


  roi_r = current_frame[right_ef[1]-dim_roi[1]:right_ef[1]+dim_roi[1], right_ef[0]-dim_roi[0]:right_ef[0]+dim_roi[0]].copy()
  roi_l = current_frame[left_ef[1]-dim_roi[1]:left_ef[1]+dim_roi[1], left_ef[0]-dim_roi[0]:left_ef[0]+dim_roi[0]].copy()


  #ellipse = cv2.fitEllipse(r)
  #cv2.ellipse(current_frame,ellipse,(0,255,0),2)

  roi_rg = cv2.cvtColor(roi_r, cv2.COLOR_BGR2GRAY)
  r = getContour(roi_rg,150)
  rb = getContour(roi_rg,110)

  roi_rb = np.zeros_like(roi_r)

  cv2.drawContours(roi_rb, [rb], -1, (0, 255, 0), 10)
  cv2.imshow('roi_rb', roi_rb)

  #cv2.drawContours(current_frame, [c], -1, (0, 255, 0), 2)

  #right = r[r[:, :, 1].argmax():,0,:]

  #, nrx, nry = findNeedle(0, right, right_ef, top, bot, right_ef[0]+dim_roi[0],  np.array([]), np.array([])) #0 right, 1 left

  
  #cv2.circle(current_frame, needle_r, 8, (0, 255, 0), -1)



  roi_lg = cv2.cvtColor(roi_l, cv2.COLOR_BGR2GRAY)
  l = getContour(roi_lg,150)

  #left = l[:l[:, :, 1].argmax()+1,0,:]




  #needle_l, nlx, nly = findCentralPoint(1, left, left_ef, top, bot, np.array([]), np.array([])) #0 right, 1 left

  #cv2.circle(current_frame, needle_l, 8, (255, 0, 255), -1)

  roi_r = np.zeros_like(roi_r)
  roi_l = np.zeros_like(roi_l)

  cv2.drawContours(roi_r, [r], -1, (0, 255, 0), 2)
  #cv2.drawContours(current_frame, [rb], -1, (0, 255, 0), 2)
  cv2.drawContours(roi_l, [l], -1, (0, 255, 255), 2)


  cv2.imshow('roi r', roi_r)
  #cv2.imshow('roi l', roi_l)

  cv2.circle(current_frame, right_ef, 8, (0, 0, 255), -1)
  cv2.circle(current_frame, left_ef, 8, (0, 255, 255), -1)
  
  #cv2.imshow('original frame', current_frame)

  diff = roi_r-roi_rb
  gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

  #d = cv2.Canny(gray,100,200)


  cv2.drawContours(diff, [], -1, (0, 255, 255), 2)

  #ellipse = cv2.fitEllipse(cnt)
  
  #print ellipse
  #cv2.ellipse(diff,ellipse,(255,255,0),2)

  cv2.imshow('diff', diff)

  cv2.waitKey(0)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  #previous_frame_gray = current_frame_gray.copy()
  previous_frame = current_frame.copy()

  grabbed, current_frame = camera.read()
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
