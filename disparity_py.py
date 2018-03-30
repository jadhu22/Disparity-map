from __future__ import print_function
from sklearn.preprocessing import normalize
import cv2
import numpy as np
import numpy as np
import argparse

def adjust_gamma(image, gamma):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	#print("hello")
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2) 

cap1.set(3,432)
cap1.set(4,240)
cap2.set(3,432)
cap2.set(4,240)

# Check if camera opened successfully
if (cap1.isOpened()== False | cap2.isOpened()==False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(1):
  # Capture frame-by-frame
  ret1, imgL = cap1.read()
  ret2, imgR = cap2.read()
  imgL = adjust_gamma(imgL, 1)
  imgR = adjust_gamma(imgR, 1)
  if (ret1 == True & ret2 == True):
      window_size = 3
      left_matcher = cv2.StereoSGBM_create(
      minDisparity=0,
      numDisparities=160,             # max_disp has to be dividable by 16 f. E. HH 192, 256
      blockSize=5,
      P1=8 * 3 * window_size ** 2,    
    	P2=32 * 3 * window_size ** 2,
    	disp12MaxDiff=1,
    	uniquenessRatio=15,
    	speckleWindowSize=0,
    	speckleRange=2,
    	preFilterCap=63,
    	mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
	)
 
      right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
 
	# FILTER Parameters
      lmbda = 80000
      sigma = 1.2
      visual_multiplier = 1.0
 
      wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
      wls_filter.setLambda(lmbda)
      wls_filter.setSigmaColor(sigma)
 
      print('computing disparity...')
      displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
      dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
      displ = np.int16(displ)
      dispr = np.int16(dispr)
      filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
 
      filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
      filteredImg = np.uint8(filteredImg)
      ret,thresh1 = cv2.threshold(filteredImg,210,255,cv2.THRESH_BINARY)

    # Display the resulting frame
      cv2.imshow('Disparity Map', filteredImg)
      #cv2.imshow('Frame1',thresh1)
    	#cv2.imshow('Frame2',imgR)
    # Press Q on keyboard to  exit
      if cv2.waitKey(1) & 0xFF == ord('q'):
      	break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap1.release()
cap2.release()
# Closes all the frames
cv2.destroyAllWindows()
