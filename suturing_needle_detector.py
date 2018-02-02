import numpy as np
import argparse
import cv2
import imutils
import fitEllipse as elps
import Queue as qe
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
stackMean = 5
meanL = [[],[]]
meanR = [[],[]]

def nothing(x):
  pass

def inMean(d,e):

  if d == 0:
    m=meanR
  else:
    m=meanL

  meanX = np.mean(m[0])
  meanY = np.mean(m[1])

  x = (e[0][0]+e[1][0])/2.0
  y = (e[0][1]+e[1][1])/2.0

  if (len(m[0])==0 or len(m[1])==0) or (x<=meanX+threshold_mean and x>=meanX-threshold_mean) and (y<=meanY+threshold_mean and y>=meanY-threshold_mean):
    if len(m[0])>=stackMean:
      del m[0][0]
    if len(m[1])>=stackMean:
      del m[1][0]
    m[0].append(x)
    m[1].append(y)
    
    return True
  return False

def getContour(img_gray,t=100):
  gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
  thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1]
  thresh = cv2.erode(thresh, None, iterations=2)
  thresh = cv2.dilate(thresh, None, iterations=2)

  a,b,c = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)

  return max(b, key=cv2.contourArea)

def sumEllipse(a,b):
  ellipse = [(0.0, 0.0), (0.0, 0.0), 0.0]
  ellipse[0]=tuple([a[0][0]+b[0][0],a[0][1]+b[0][1]])
  ellipse[1]=tuple([a[1][0]+b[1][0],a[1][1]+b[1][1]])
  ellipse[2]=a[2]+b[2]

  return ellipse

def diffEllipse(a,b):
  ellipse = [(0.0, 0.0), (0.0, 0.0), 0.0]
  ellipse[0]=tuple([abs(a[0][0]-b[0][0]),abs(a[0][1]-b[0][1])])
  ellipse[1]=tuple([abs(a[1][0]-b[1][0]),abs(a[1][1]-b[1][1])])
  ellipse[2]=abs(a[2]-b[2])

  return ellipse

'''
def lessThan(a,b):
  ma = tuple([(a[0][0]+a[1][0])/2,(a[0][1]+a[1][1])/2])
  mb = tuple([(b[0][0]+b[1][0])/2,(b[0][1]+b[1][1])/2])


def nearestEllipse(e,l):
  near = [(0.0, 0.0), (0.0, 0.0), 0.0]
  diff = [(1000000.0, 1000000.0), (1000000.0, 1000000.0), 1000000.0]
  for i in range(len(l)):
    if lessThan(diffEllipse(l[i],e),diff):
      diff = diffEllipse(l[i],e)
      near = l[i]
  return near
'''

def meanEllipse(l):
  bestList = 0
  numEllipse = 0

  for i in range(len(l)):
    numEllipse += len(l[i])
    if len(l[i])>bestList:
      bestList=i

  if len(l)==0 or numEllipse<threshold_ellipse:
    return None

  ellipse = [(0.0, 0.0), (0.0, 0.0), 0.0]
  n = len(l[bestList])

  for i in range(n):
    ellipse=sumEllipse(ellipse,l[bestList][i])

  ellipse[0]=tuple([ellipse[0][0]/n,ellipse[0][1]/n])
  ellipse[1]=tuple([ellipse[1][0]/n,ellipse[1][1]/n])
  ellipse[2]=ellipse[2]/n
  ellipse = tuple(ellipse)

  return ellipse

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

cv2.namedWindow('ellipses_thresholds',)
cv2.createTrackbar('roi_dimension','ellipses_thresholds',dim_roi[0],150,nothing)
cv2.createTrackbar('upper_min','ellipses_thresholds',upper[0],upper[1],nothing)
cv2.createTrackbar('upper_MAX','ellipses_thresholds',upper[1],250,nothing)
cv2.createTrackbar('lower_min','ellipses_thresholds',lower[0],lower[1],nothing)
cv2.createTrackbar('lower_MAX','ellipses_thresholds',lower[1],100,nothing)

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

  dim_roi[0],dim_roi[1] = cv2.getTrackbarPos('roi_dimension', 'ellipses_thresholds'),cv2.getTrackbarPos('roi_dimension', 'ellipses_thresholds')
  upper = [cv2.getTrackbarPos('upper_min', 'ellipses_thresholds'),cv2.getTrackbarPos('upper_MAX', 'ellipses_thresholds')]
  lower = [cv2.getTrackbarPos('lower_min', 'ellipses_thresholds'),cv2.getTrackbarPos('lower_MAX', 'ellipses_thresholds')]

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

  roi_r = current_frame[abs(right_ef[1]-dim_roi[1]):right_ef[1]+dim_roi[1], abs(right_ef[0]-dim_roi[0]):right_ef[0]+dim_roi[0]].copy()
  roi_l = current_frame[abs(left_ef[1]-dim_roi[1]):left_ef[1]+dim_roi[1], abs(left_ef[0]-dim_roi[0]):left_ef[0]+dim_roi[0]].copy()

  min_window = [100,100]
  offset = 20
  dx = max(abs(left_ef[0]-right_ef[0]),min_window[0])+offset
  dy = max(abs(left_ef[1]-right_ef[1]),min_window[1])+offset
  roi_middle = current_frame[left_ef[1]-offset:left_ef[1]+dy, left_ef[0]-offset:left_ef[0]+dx].copy()
  cv2.imshow('middle',roi_middle)

  #COMPUTE GRAYSCALE AND CONTOUR
  roi_rg = cv2.cvtColor(roi_r, cv2.COLOR_BGR2GRAY)
  r = getContour(roi_rg,150)

  roi_lg = cv2.cvtColor(roi_l, cv2.COLOR_BGR2GRAY)
  l = getContour(roi_lg,150)
  

  roi_middle_gray = cv2.cvtColor(roi_middle, cv2.COLOR_BGR2GRAY)
  m = getContour(roi_middle_gray,150)


  #cv2.drawContours(current_frame[abs(right_ef[1]-dim_roi[1]):right_ef[1]+dim_roi[1], abs(right_ef[0]-dim_roi[0]):right_ef[0]+dim_roi[0]], [r], -1, (255, 0, 0), 2)
  #cv2.drawContours(current_frame[abs(left_ef[1]-dim_roi[1]):left_ef[1]+dim_roi[1], abs(left_ef[0]-dim_roi[0]):left_ef[0]+dim_roi[0]], [l], -1, (0, 255, 0), 2)
  cv2.drawContours(current_frame[left_ef[1]-offset:left_ef[1]+dy, left_ef[0]-offset:left_ef[0]+dx], [m], -1, (0, 255, 0), 2)

  roi_r = np.zeros_like(roi_r)
  roi_l = np.zeros_like(roi_l)

  lefta = l[:l[:, :, 1].argmax()+1,0,:]
  righto = r[r[:, :, 1].argmax():,0,:]

  size = 40
  step = 10
  ellipses_l = []
  ellipses_r = []
  stepList = []

  bkl = roi_l.copy()

  #LEFT
  for i in range(len(lefta) - size):
    a = lefta[i:i+size]
    ellipse = cv2.fitEllipse(a)
    cv2.drawContours(bkl, [l], -1, (0, 255, 0), 2)
    cv2.drawContours(bkl, [a], -1, (255, 0, 0), 2)

    if i%step==0:
      if len(stepList)!=0:
        ellipses_l.append(stepList)
      stepList = []
    if inMean(0,ellipse):  
      if (ellipse[2]>upper[0]):# and ellipse[2]<upper[1]):# or (ellipse[2]>lower[0] and ellipse[2]<lower[1]):
        stepList.append(ellipse)
        cv2.ellipse(bkl,ellipse,(0,0,255),2)
      else:
        cv2.ellipse(bkl,ellipse,(255,0,0),2)

    cv2.imshow('bkl', bkl)
    bkl = roi_l.copy()
    #cv2.waitKey(0)

  bkr = roi_r.copy()
  
  #RIGHT
  for i in range(len(righto) - size):
    a = righto[i:i+size]
    ellipse = cv2.fitEllipse(a)
    cv2.drawContours(bkr, [r], -1, (0, 255, 0), 2)
    cv2.drawContours(bkr, [a], -1, (255, 0, 0), 2)

    if i%step==0:
      if len(stepList)!=0:
        ellipses_r.append(stepList)
      stepList = []
    
    if inMean(1,ellipse):  
      if (ellipse[2]>upper[0]):# and ellipse[2]<upper[1]):# or (ellipse[2]>lower[0] and ellipse[2]<lower[1]):
        stepList.append(ellipse)
        cv2.ellipse(bkr,ellipse,(0,0,255),2)
      else:
        cv2.ellipse(bkr,ellipse,(255,0,0),2)

    cv2.imshow('bkr', bkr)
    bkr = roi_r.copy()
    #cv2.waitKey(0)


  mediumEllipseL = meanEllipse(ellipses_l)
  mediumEllipseR = meanEllipse(ellipses_r)
  if mediumEllipseL and mediumEllipseR:
    if mediumEllipseR[2]>mediumEllipseL[2]:
      cv2.ellipse(current_frame[abs(right_ef[1]-dim_roi[1]):right_ef[1]+dim_roi[1], abs(right_ef[0]-dim_roi[0]):right_ef[0]+dim_roi[0]],mediumEllipseR,(0,0,255),2)
    else:
      cv2.ellipse(current_frame[abs(left_ef[1]-dim_roi[1]):left_ef[1]+dim_roi[1], abs(left_ef[0]-dim_roi[0]):left_ef[0]+dim_roi[0]],mediumEllipseL,(0,255,255),2)
  elif mediumEllipseL:
    cv2.ellipse(current_frame[abs(left_ef[1]-dim_roi[1]):left_ef[1]+dim_roi[1], abs(left_ef[0]-dim_roi[0]):left_ef[0]+dim_roi[0]],mediumEllipseL,(0,255,255),2)
  elif mediumEllipseR:
    cv2.ellipse(current_frame[abs(right_ef[1]-dim_roi[1]):right_ef[1]+dim_roi[1], abs(right_ef[0]-dim_roi[0]):right_ef[0]+dim_roi[0]],mediumEllipseR,(0,0,255),2)



  cv2.circle(current_frame, right_ef, 8, (0, 0, 255), -1)
  cv2.circle(current_frame, left_ef, 8, (0, 255, 255), -1)
  cv2.imshow('original frame', current_frame)
  cv2.waitKey(1)

  #elif cv2.waitKey(1) & 0xFF == ord('q'):
  #  break
 


  #previous_frame_gray = current_frame_gray.copy()
  previous_frame = current_frame.copy()

  grabbed, current_frame = camera.read()
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
