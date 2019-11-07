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
from scipy.optimize import fsolve


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
    # initialize a publisher to send joints' angular position to the robot
    self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
    self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
    self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()
    #scale (projection in plane parallel to camera through yellow blob) determined for all angles=0
    

  #___________________detection of the blobs__________________________
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

  #______________get the projection from published blob data______________
  def get_projection(self,blobs,r_yellow,scale):
    pos_cam = np.array(blobs).reshape((4,2))
    pos_cam = scale*pos_cam
    pos_cam[0,1] = pos_cam[0,1]+r_yellow #takes into account that only half yellow blob is visible
    pos_cam = pos_cam-pos_cam[0]
    pos_cam[:,1]=-pos_cam[:,1]
    return pos_cam

  #__________________matrix calculation for green blob __________________
  #position of green blob is x,y,z without rotation around z axis (yellow blob) and then
  #rotated by a rotation matrix around z: rot_z(theta1)*xyz(theta2,theta3)
  def pos_green_blob(self,theta1,theta2,theta3):
    x = 3*np.sin(theta3)
    y = -3*np.sin(theta2)*np.cos(theta3)
    z = 2+3*np.cos(theta2)*np.cos(theta3)
    rot = np.array([[np.cos(theta1),-np.sin(theta1),0],
    		[np.sin(theta1),np.cos(theta1),0],
    		[0,0,1]])

    return rot.dot(np.array([x,y,z]))


  # _______________Recieve data, process it, and publish______________________
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



  #_________________________combine both images____________________________
  def callbackmaster(self,data):
    #save the projection into a matrix
    pos_cam1 = self.get_projection(data.data,0.43,5/134.)
    pos_cam2 = self.get_projection(self.blob_pos2.data,0.3,5/132.)

    #get the three-dimensional point vector of the green blob
    x_measured = np.array([pos_cam2[2,0],pos_cam1[2,0],pos_cam2[2,1]]) #x,y,z coordinates from different cameras

    #define the function for fsolve (numerical solver for the angles given the measured position of the green blob)
    def function_for_fsolve(theta):
	return np.array([3*(np.cos(theta[0])*np.sin(theta[2])+np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]))-x_measured[0],
			3*(np.sin(theta[0])*np.sin(theta[2])-np.cos(theta[0])*np.sin(theta[1])*np.cos(theta[2]))-x_measured[1],
			2+3*np.cos(theta[1])*np.cos(theta[2])-x_measured[2]])
    #perform solver
    theta_est = fsolve(function_for_fsolve,np.array([0,0,0]),xtol=1e-3)

    
    #define desired joint angles
    q_d = [0,np.pi/2,0]		#move robot here
    self.joint1=Float64()
    self.joint1.data= q_d[0]
    self.joint2=Float64()
    self.joint2.data= q_d[1]
    self.joint3=Float64()
    self.joint3.data= q_d[2]
    
    print("pos out of measured angle:\t{}".format(function_for_fsolve(theta_est)+x_measured))
    print("measured pos:\t\t\t{}".format(x_measured))
    print("theoretical position:\t\t{}\n".format(self.pos_green_blob(*q_d)))
    print("measured angle:\t\t\t{}".format(np.fmod(theta_est,2*np.pi)))
    print("desired angle:\t\t\t{}\n\n".format(q_d))
    
    #publish results
    try: 
      self.robot_joint1_pub.publish(self.joint1)
      self.robot_joint2_pub.publish(self.joint2)
      self.robot_joint3_pub.publish(self.joint3)
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


