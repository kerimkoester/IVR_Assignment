#!/usr/bin/env python

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
import math


class image_converter:

  # Defines publisher and subscriber
  def __init__(self):
    # initialize the node named image_processing
    rospy.init_node('image1', anonymous=True)
    # initialize a publisher to send images from camera1 to a topic named image_topic1
    self.image_pub1 = rospy.Publisher("image_topic1",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)
    # initialize a publisher to publish position of blobs
    self.blob_pub1 = rospy.Publisher("/camera1/blob_pos",Float64MultiArray, queue_size=10)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()
    
  def detect_red(self,image):
      # Isolate the blue colour in the image as a binary image
      mask = cv2.inRange(image, (0, 0, 100), (0, 0, 255))
      # This applies a dilate that makes the binary region larger (the more iterations the larger it becomes)
      kernel = np.ones((5, 5), np.uint8)
      mask = cv2.dilate(mask, kernel, iterations=3)
      # Obtain the moments of the binary image
      M = cv2.moments(mask)
      # Calculate pixel coordinates for the centre of the blob
      if M['m00'] == 0:
	return np.array([np.nan,np.nan])
      cx = int(M['m10'] / M['m00'])
      cy = int(M['m01'] / M['m00'])
      return np.array([cx, cy])
  # Detecting the centre of the green circle
  def detect_green(self,image):
      mask = cv2.inRange(image, (0, 100, 0), (0, 255, 0))
      kernel = np.ones((5, 5), np.uint8)
      mask = cv2.dilate(mask, kernel, iterations=3)
      M = cv2.moments(mask)
      if M['m00'] == 0:
	return np.array([np.nan,np.nan])
      cx = int(M['m10'] / M['m00'])
      cy = int(M['m01'] / M['m00'])
      return np.array([cx, cy])
  # Detecting the centre of the blue circle
  def detect_blue(self,image):
      mask = cv2.inRange(image, (100, 0, 0), (255, 0, 0))
      kernel = np.ones((5, 5), np.uint8)
      mask = cv2.dilate(mask, kernel, iterations=3)
      M = cv2.moments(mask)
      if M['m00'] == 0:
	return np.array([np.nan,np.nan])
      cx = int(M['m10'] / M['m00'])
      cy = int(M['m01'] / M['m00'])
      return np.array([cx, cy])
  # Detecting the centre of the yellow circle
  def detect_yellow(self,image):
      mask = cv2.inRange(image, (0, 100, 100), (0, 255, 255))
      kernel = np.ones((5, 5), np.uint8)
      mask = cv2.dilate(mask, kernel, iterations=3)
      M = cv2.moments(mask)
      if M['m00'] == 0:
	return np.array([np.nan,np.nan])
      cx = int(M['m10'] / M['m00'])
      cy = int(M['m01'] / M['m00'])
      return np.array([cx, cy])

 #________________Target Detection______________________________________________


  def detectTarget(self, sourceImg):
    #segment target using color threshold
    targetImg = cv2.inRange(sourceImg,(10,10,120),(100,255,255))
    
    #get contours of the orange objects
    img = cv2.bitwise_not(targetImg)
    contours, hierarchy = cv2.findContours(img, 1, 2)

    self.cnt1 = contours[0]
    self.cnt2 = contours[1]

    #display the contour of a sphere
    if isinstance(self.compareCnts(),float):
      return np.array([-1.5,-1.5])
    else:
      self.sphere = cv2.inRange(sourceImg,(100,100,100),(101,101,101))
      cv2.drawContours(self.sphere,[self.compareCnts()],0,(255,0,255),1)
      return self.findCenter(self.sphere)
    

  #calculate the circularity of each contour
  def getCircularity(self,cnt):
    perimeter = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt)
    if perimeter == 0:
      return np.nan
    else:
      return (4*math.pi*area)/(perimeter**2)

  #compares two contour and returns the one that is less circular
  def compareCnts(self):
    if np.isnan(self.getCircularity(self.cnt1)) or np.isnan(self.getCircularity(self.cnt2)):
      return np.nan
    if self.getCircularity(self.cnt1)>self.getCircularity(self.cnt2):
      return self.cnt1
    else:
      return self.cnt2

  #findCenter of the contour
  def findCenter(self, cnt):
    M = cv2.moments(cnt)
    if M['m00'] == 0:
      return np.array([np.nan,np.nan])
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
   
    return [cx,cy]




  # Recieve data from camera 1, process it, and publish
  def callback1(self,data):
    # Recieve the image
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    #im1=cv2.imshow('window1', self.cv_image1)
    #cv2.waitKey(1)
    self.orange_sphere1 = self.detectTarget(self.cv_image1)
    self.blob_pos1=Float64MultiArray()
    self.blob_pos1.data=np.array([self.detect_yellow(self.cv_image1),self.detect_blue(self.cv_image1),
				  self.detect_green(self.cv_image1),self.detect_red(self.cv_image1),self.orange_sphere1]).flatten()

    #cv2.imwrite('cam1.png',self.cv_image1)
    #Publish the results
    try: 
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
      self.blob_pub1.publish(self.blob_pos1)
    except CvBridgeError as e:
      print(e)

# call the class
def main(args):
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)


