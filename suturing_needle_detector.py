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

(grabbed, current_frame) = camera.read()
previous_frame = current_frame

while camera != 0:

  if not grabbed:
    break

  current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
  previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

  frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)

  '''
  ret, threshold_output = cv2.threshold(frame_diff, 127, 255, 0)
  im2, contours, hierarchy = cv2.findContours(threshold_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  
  minEllipse = np.array(len(contours))

  for i in range(len(contours)):
    if len(contours[i]) > 5:
      minEllipse[i] = cv2.fitEllipse(contours[i])
  '''

  # Draw contours + rotated rects + ellipses
  #drawing = np.zeros(len(threshold_output), cv2.CV_8UC3)

  '''for( int i = 0; i< contours.size(); i++ )
     {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       // contour
       drawContours( drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
       // ellipse
       ellipse( drawing, minEllipse[i], color, 2, 8 );
       // rotated rectangle
       Point2f rect_points[4]; minRect[i].points( rect_points );
       for( int j = 0; j < 4; j++ )
          line( drawing, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );
     }
  
  cv2.drawContours(current_frame, contours, -1, (0, 255, 0), 3)
  '''


  cv2.imshow('frame diff ', frame_diff)
  cv2.imshow('original frame ', current_frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  previous_frame = current_frame.copy()

  grabbed, current_frame = camera.read()

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
