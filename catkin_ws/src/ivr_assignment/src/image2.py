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


class image_converter:

  # Defines publisher and subscriber
  def __init__(self):
    # initialize the node named image_processing
    rospy.init_node('image2', anonymous=True)
    # initialize a publisher to send images from camera2 to a topic named image_topic2
    self.image_pub2 = rospy.Publisher("image_topic2",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw",Image,self.callback2)
    self.image1_sub = rospy.Subscriber("/camera1/blob_pos",Float64MultiArray,self.callbackmaster)
    # initialize a publisher to publish position of blobs
    self.blob_pub2 = rospy.Publisher("/camera2/blob_pos",Float64MultiArray, queue_size=10)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()
    #scale (projection in plane parallel to camera through yellow blob) determined for all angles=0
    
  def detect_red(self,image):
      # Isolate the blue colour in the image as a binary image
      mask = cv2.inRange(image, (0, 0, 100), (0, 0, 255))
      # This applies a dilate that makes the binary region larger (the more iterations the larger it becomes)
      kernel = np.ones((5, 5), np.uint8)
      mask = cv2.dilate(mask, kernel, iterations=3)
      # Obtain the moments of the binary image
      M = cv2.moments(mask)
      # Calculate pixel coordinates for the centre of the blob
      cx = int(M['m10'] / M['m00'])
      cy = int(M['m01'] / M['m00'])
      return np.array([cx, cy])

  # Detecting the centre of the green circle
  def detect_green(self,image):
      mask = cv2.inRange(image, (0, 100, 0), (0, 255, 0))
      kernel = np.ones((5, 5), np.uint8)
      mask = cv2.dilate(mask, kernel, iterations=3)
      M = cv2.moments(mask)
      cx = int(M['m10'] / M['m00'])
      cy = int(M['m01'] / M['m00'])
      return np.array([cx, cy])

  # Detecting the centre of the blue circle
  def detect_blue(self,image):
      mask = cv2.inRange(image, (100, 0, 0), (255, 0, 0))
      kernel = np.ones((5, 5), np.uint8)
      mask = cv2.dilate(mask, kernel, iterations=3)
      M = cv2.moments(mask)
      cx = int(M['m10'] / M['m00'])
      cy = int(M['m01'] / M['m00'])
      return np.array([cx, cy])

  # Detecting the centre of the yellow circle
  def detect_yellow(self,image):
      mask = cv2.inRange(image, (0, 100, 100), (0, 255, 255))
      kernel = np.ones((5, 5), np.uint8)
      mask = cv2.dilate(mask, kernel, iterations=3)
      M = cv2.moments(mask)
      cx = int(M['m10'] / M['m00'])
      cy = int(M['m01'] / M['m00'])
      return np.array([cx, cy])

  def get_projection(self,blobs,r_yellow,scale):
    pos_cam = np.array(blobs).reshape((4,2))
    pos_cam = scale*pos_cam
    pos_cam[0,1] = pos_cam[0,1]+r_yellow #takes into account that only half yellow blob is visible
    pos_cam = pos_cam-pos_cam[0]
    pos_cam[:,1]=-pos_cam[:,1]
    return pos_cam
  # Recieve data, process it, and publish
  def callback2(self,data):
    # Recieve the image
    try:
      self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    
    self.blob_pos2=Float64MultiArray()
    self.blob_pos2.data=np.array([self.detect_yellow(self.cv_image2),self.detect_blue(self.cv_image2),self.detect_green(self.cv_image2),self.detect_red(self.cv_image2)]).flatten()
    im2=cv2.imshow('window2', self.cv_image2)
    cv2.waitKey(1)

    # Publish the results
    try: 
      self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
      self.blob_pub2.publish(self.blob_pos2)
    except CvBridgeError as e:
      print(e)
      
  def callbackmaster(self,data):
    pos_cam1 = self.get_projection(data.data,0.43,5/134.)
    pos_cam2 = self.get_projection(self.blob_pos2.data,0.3,5/132.)
    def rotx(alpha):
        return np.array([[1,0,0,0],
                                        [0,np.cos(alpha),-np.sin(alpha),0],
                                        [0,np.sin(alpha),np.cos(alpha),0],
                                        [0,0,0,1]])
    def rotz(alpha):
        return np.array([  [np.cos(alpha),-np.sin(alpha),0,0],
                                        [np.sin(alpha),np.cos(alpha),0,0],
                                        [0,0,1,0],
                                        [0,0,0,1]])
    def trans_x(a):
        return np.array([[1,0,0,0],
                                     [0,1,0,0],
                                     [0,0,1,a],
                                     [0,0,0,1]])
    def trans(alpha,d,theta):
        return rotz(theta).dot(trans_x(d).dot(rotx(alpha)))
    
    def trans_tot(theta1,theta2,theta3):
        return trans(-np.pi/2,2,theta1).dot(trans(np.pi/2,0,theta2).dot(trans(np.pi/2,3,theta3)))
    
    print("transmat: ",trans_tot(0,0,0)[:,3])
        
    print("y,z: ",pos_cam1)
    print("x,z: ",pos_cam2)

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


