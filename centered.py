import numpy as np
import argparse
import cv2
#import imutils
import fitEllipse as elps
#import Queue as qe

img_size = [640,480]
threshold_px = 100
dim_roi = [80,80]
upper=[160,180]
lower=[10,20]
threshold_ellipse = 10
threshold_mean=0
stackMean = 20
mean_list = [[],[],[]] #center, width, height
min_window = [80,80]
max_window = [200,200]
offset = 30
ellipse_size = 40
primo = 1
prevEd = 0  
nframe = 0

def nothing(x):
  pass

def nearPoint(a,b,th=0): #a in b+-th
  if(type(b)==float):
    if a[0]<=b+th and a[0]>=b-th and a[1]<=b+th and a[1]>=b-th:
      return True
  elif a[0]<=b[0]+th and a[0]>=b[0]-th and a[1]<=b[1]+th and a[1]>=b[1]-th:
    return True
  return False

def findMostSimilar(ellipse,l, threshold=10):
  best = [(0.0, 0.0), (0.0, 0.0), 500.0]
  best_selected = []

  for e in l: #finde the most similar in position
    if nearPoint(e[0],ellipse[0],threshold):
      best_selected.append(e)

  index_list = []
  for i in range(len(best_selected)): #finde the most similar in dimension
    if not(nearPoint(best_selected[i][1],ellipse[1],threshold)):
      index_list.append(i)

  for i in range(len(best_selected)): #finde the most similar in angle
    if i in index_list:
      continue
    if abs(ellipse[2]-e[2])<best[2]:
      best = e

  if best == [(0.0, 0.0), (0.0, 0.0), 500.0]:
    return ellipse

  return best 

def inMean(e):
  
  center = e[0]
  width = e[1][0]
  height = e[1][1]
  phi = e[2]

  if len(mean_list[0]) < stackMean and len(mean_list[1]) < stackMean and len(mean_list[2]) < stackMean:
    mean_list[0].append(center)
    mean_list[1].append(width)   
    mean_list[2].append(height) 
    return True

  meanC = np.mean(mean_list[0])
  meanW = np.mean(mean_list[1])
  meanH = np.mean(mean_list[2])

  print ('c ', meanC)
  print ('w ', meanW)
  print ('h ', meanH)
  print (nearPoint(center,meanC,threshold_mean))

  if nearPoint(center,meanC,threshold_mean) \
    and (width<=meanW+threshold_mean and width>=meanW-threshold_meanth) \
    and (height<=meanH+threshold_mean and height>=meanH-threshold_mean):

    if len(mean_list[0])>=stackMean:
      del mean_list[0][0]
    if len(mean_list[1])>=stackMean:
      del mean_list[1][0]
    if len(mean_list[2])>=stackMean:
      del mean_list[2][0]

    mean_list[0].append(center)
    mean_list[1].append(width)   
    mean_list[2].append(height) 
    
    return True
  return False

def getContour(img_gray,t=100):
  gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
  thresh = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)[1]
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

def meanEllipse(l):
  bestList = 0
  numEllipse = 0

  for i in range(len(l)):
    numEllipse += len(l[i])
    if len(l[i])>len(l[bestList]):
      bestList=i

  if len(l)==0 or numEllipse<threshold_ellipse:
    return None, None

  ellipse = [(0.0, 0.0), (0.0, 0.0), 0.0]
  n = len(l[bestList])

  for i in range(n):
    ellipse=sumEllipse(ellipse,l[bestList][i])

  ellipse[0]=tuple([ellipse[0][0]/n,ellipse[0][1]/n])
  ellipse[1]=tuple([ellipse[1][0]/n,ellipse[1][1]/n])
  ellipse[2]=ellipse[2]/n
  ellipse = tuple(ellipse)

  return ellipse, bestList

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


camera = cv2.VideoCapture("/media/francesco/UbuntuHD/medical_robotics/Suturing/video/Suturing_B001_capture2.avi")
camera.set(cv2.CAP_PROP_FRAME_WIDTH, img_size[0])
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, img_size[1])
camera.set(cv2.CAP_PROP_FPS, 1)

print("opencv version "+cv2.__version__)

cv2.namedWindow('ellipses_thresholds',)
cv2.createTrackbar('ellipse_size','ellipses_thresholds',ellipse_size,150,nothing)
'''cv2.createTrackbar('upper_min','ellipses_thresholds',upper[0],upper[1],nothing)
cv2.createTrackbar('upper_MAX','ellipses_thresholds',upper[1],250,nothing)
cv2.createTrackbar('lower_min','ellipses_thresholds',lower[0],lower[1],nothing)
cv2.createTrackbar('lower_MAX','ellipses_thresholds',lower[1],100,nothing)
'''
for i in range(900):
  (grabbed, current_frame) = camera.read()
  nframe = i

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
  nframe = nframe + 1
  #print (nframe)
  ellipse_size_bk = ellipse_size
  ellipse_size = cv2.getTrackbarPos('ellipse_size', 'ellipses_thresholds')
  if ellipse_size != ellipse_size_bk:
    mean_list = [[],[]]

  #upper = [cv2.getTrackbarPos('upper_min', 'ellipses_thresholds'),cv2.getTrackbarPos('upper_MAX', 'ellipses_thresholds')]
  #lower = [cv2.getTrackbarPos('lower_min', 'ellipses_thresholds'),cv2.getTrackbarPos('lower_MAX', 'ellipses_thresholds')]

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

  dx = abs(left_ef[0]-right_ef[0])
  dy = abs(left_ef[1]-right_ef[1])

  if(dx > min_window[0]):
    dx = 0
  else:
    dx = min_window[0]

  if(dy > min_window[1]):
    dy = 0
  else:
    dy = min_window[1] 


  
  #print ("dx ",dx,"dy ",dy)

  if(left_ef[0] >= right_ef[0]):
    if(left_ef[1] <= right_ef[1]):
      print ("left ",left_ef, " right ",right_ef)
      centered_roi_pointer = current_frame[max(left_ef[1]-(dy/2),0):right_ef[1]+(dy/2), max(right_ef[0]-(dx/2),0):left_ef[0]+(dx/2)]
    else:
      centered_roi_pointer = current_frame[max(right_ef[1]-(dy/2),0):left_ef[1]+(dy/2), max(right_ef[0]-(dx/2),0):left_ef[0]+(dx/2)]
  else:
    if(left_ef[1] <= right_ef[1]):
      centered_roi_pointer = current_frame[max(left_ef[1]-(dy/2),0):right_ef[1]+(dy/2), max(left_ef[0]-(dx/2),0):right_ef[0]+(dx/2)]
    else:
      centered_roi_pointer = current_frame[max(right_ef[1]-(dy/2),0):left_ef[1]+(dy/2), max(left_ef[0]-(dx/2),0):right_ef[0]+(dx/2)]

  
  #print ("Y diff ",(left_ef[1]-(dy/2)) - (right_ef[1]+(dy/2)))

  #print ("X diff " ,(left_ef[0]-(dx/2))-(right_ef[0]+(dx/2)))
  


  centered_roi = centered_roi_pointer.copy()

  #COMPUTE GRAYSCALE AND CONTOUR  
  
  centered_roi_gray = cv2.cvtColor(centered_roi, cv2.COLOR_BGR2GRAY)
  m = getContour(centered_roi_gray,150)

  cv2.drawContours(centered_roi, [m], -1, (0, 255, 0), 2)

  pixels = m[:,0,:]

  step = 10
  ellipses = []
  stepList = []

  bk = centered_roi.copy()

  frame_test_bk = current_frame.copy()
  centered_roi_pointer_test_bk = frame_test_bk[left_ef[1]-offset:left_ef[1]+dy, left_ef[0]-offset:left_ef[0]+dx]

  for i in range(len(pixels) - ellipse_size):
    a = pixels[i:i+ellipse_size]
    
    ellipse = cv2.fitEllipse(a)
    cv2.drawContours(bk, [m], -1, (0, 255, 0), 2)
    cv2.drawContours(bk, [a], -1, (255, 0, 0), 2)

    
    if i%step==0:
      if len(stepList)!=0:
        ellipses.append(stepList)
      stepList = []
     
    if (ellipse[1][0]<=ellipse_size or ellipse[1][1]<=ellipse_size):
      #if inMean(ellipse):  
       # if (ellipse[2]>upper[0]):# and ellipse[2]<upper[1]):# or (ellipse[2]>lower[0] and ellipse[2]<lower[1]):
        #  stepList.append(ellipse)

      stepList.append(ellipse)
      cv2.ellipse(bk, ellipse, (0,255,0), 2)
      #cv2.ellipse(bk, ell, (0,0,255), 2) 

      cv2.ellipse(centered_roi_pointer_test_bk, ellipse, (255,0,0), 2)

      #cv2.ellipse(centered_roi_pointer,ellipse,(0,255,0),2)

      #  else:
       #   cv2.ellipse(bk,ellipse,(255,0,0),2)

    
    cv2.imshow('bk', bk)

    #cv2.waitKey(0)
    bk = centered_roi.copy()

  mediumEllipse, best_list = meanEllipse(ellipses)
  
  if(primo == 1):
    prevEll = 0
  
 
  if mediumEllipse:
    cv2.ellipse(centered_roi_pointer,mediumEllipse,(255,0,0),2)
    mediumEllipse = findMostSimilar(mediumEllipse,ellipses[best_list], 20)
    if(primo != 1):
      prevEll = mediumEllipse
      prevEd = 1
      vec = [abs(prevEll[0][0] - mediumEllipse[0][0]),abs(prevEll[0][1] - mediumEllipse[0][1])]
      res = np.linalg.norm(vec)
      if(res < 30):
        
        cv2.ellipse(centered_roi_pointer,mediumEllipse,(0,0,255),2)
      else:
        cv2.ellipse(centered_roi_pointer,prevEll,(0,0,255),2)
      
     
    else: 
      primo = 0
   
  else:
    
    if (prevEd == 1):
      cv2.ellipse(centered_roi_pointer,prevEll,(0,0,255),2)


  cv2.circle(current_frame, left_ef, 8, (0, 255, 255), -1)
  cv2.circle(current_frame, right_ef, 8, (70, 255, 135), -1)
  cv2.imshow('original frame', current_frame)
  cv2.imshow('all ellipses', frame_test_bk)
  cv2.moveWindow("original frame", 1200,0)
  cv2.moveWindow("all ellipses", 500,500)

  cv2.waitKey(1)

  #elif cv2.waitKey(1) & 0xFF == ord('q'):
  #  break
 


  #previous_frame_gray = current_frame_gray.copy()
  previous_frame = current_frame.copy()

  grabbed, current_frame = camera.read()
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
