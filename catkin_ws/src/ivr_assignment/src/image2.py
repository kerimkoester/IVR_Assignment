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
    self.bridge = CvBridge()
    #scale (projection in plane parallel to camera through yellow blob) determined for all angles=0
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
    self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
    # initialize the bridge between openCV and ROS
    

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

  #______________get the projection from published blob data______________
  def eliminate_nonvisible_blobs(self,blobs):    
    if np.isnan(blobs[4]):
      blobs[4]=blobs[2]
      blobs[5]=blobs[3]
    if np.isnan(blobs[6]):
      blobs[6]=blobs[4]
      blobs[7]=blobs[5]
    return blobs

  '''def get_projection1(self,blobs,r_yellow,scale):
    corrected_blobs = self.eliminate_nonvisible_blobs(blobs)
    pos_cam = np.array(corrected_blobs).reshape((4,2))
    pos_cam = scale*pos_cam
    pos_cam[0,1] = pos_cam[0,1]+r_yellow #takes into account that only half yellow blob is visible
    pos_cam = pos_cam-pos_cam[0]
    pos_cam[:,1]=-pos_cam[:,1]
    return np.array(pos_cam)'''

  def get_projection(self,blobs,scale):
    corrected_blobs = self.eliminate_nonvisible_blobs(blobs)
    pos_cam = np.array(corrected_blobs).reshape((4,2))
    pos_cam = scale*pos_cam
    pos_cam = pos_cam-pos_cam[1]
    pos_cam[:,1]=-pos_cam[:,1]
    return np.array(pos_cam)

  #________________use projection to get blob position___________________
  
  #perform a weighted average over the two z-measurements
  #blobs closer to the camera are more distorted than blobs farer away. Use the z-value from blob which is farer away
  def z_average(self,z1,z2,x,y):
    w1 = (5-x)**2
    w2 = (5+y)**2
    if x==-y:
      return (z1+z2)/2
    else:
      return (w1*z1+w2*z2)/(w1+w2)
  
  def yellow_blob_measured(self,cam1,cam2):
    return np.array([cam2[0,0],cam1[0,0],self.z_average(cam1[0,1],cam2[0,1],cam2[0,0],cam1[0,0])])
  def blue_blob_measured(self,cam1,cam2):
    return np.array([cam2[1,0],cam1[1,0],self.z_average(cam1[1,1],cam2[1,1],cam2[1,0],cam1[1,0])])
  def green_blob_measured(self,cam1,cam2):
    return np.array([cam2[2,0],cam1[2,0],self.z_average(cam1[2,1],cam2[2,1],cam2[2,0],cam1[2,0])])
  def red_blob_measured(self,cam1,cam2):
    return np.array([cam2[3,0],cam1[3,0],self.z_average(cam1[3,1],cam2[3,1],cam2[3,0],cam1[3,0])])
  def blobs_measured(self,cam1,cam2):
    return np.array([self.yellow_blob_measured(cam1,cam2),self.blue_blob_measured(cam1,cam2),
		     self.green_blob_measured(cam1,cam2),self.red_blob_measured(cam1,cam2)])
    

  #__________________matrix calculation for green blob __________________
  #position of green blob is x,y,z without rotation around z axis (yellow blob) and then
  #rotated by a rotation matrix around z: rot_z(theta1)*xyz(theta2,theta3)
  def pos_green_blob(self,theta1,theta2,theta3):
    x = 3*np.sin(theta3)
    y = -3*np.sin(theta2)*np.cos(theta3)
    z = 3*np.cos(theta2)*np.cos(theta3)
    rot = np.array([[np.cos(theta1),-np.sin(theta1),0],
    		[np.sin(theta1),np.cos(theta1),0],
    		[0,0,1]])

    return rot.dot(np.array([x,y,z]))

  #_____________rotation-matrix for red blob______________________
  def rotz(self,theta):
    return np.array([[np.cos(theta),-np.sin(theta),0],
	  	   [np.sin(theta),np.cos(theta),0],
	  	   [0,0,1]])
  def rotx(self,theta):
    return np.array([[1,0,0],
	  	   [0,np.cos(theta),-np.sin(theta)],
	  	   [0,np.sin(theta),np.cos(theta)]])
  def roty(self,theta):
    return np.array([[np.cos(theta),0,np.sin(theta)],
	  	   [0,1,0],
	  	   [-np.sin(theta),0,np.cos(theta)]])
  def rot_tot(self,theta1,theta2,theta3,theta4):
    return self.rotz(theta1).dot(self.rotx(theta2).dot(self.roty(theta3).dot(self.rotx(theta4))))
  def rot_123(self,theta1,theta2,theta3):
    return self.rotz(theta1).dot(self.rotx(theta2).dot(self.roty(theta3)))
  
  def pos_red_blob(self,green_blob,theta1,theta2,theta3,theta4):
    return green_blob+2*self.rot_tot(theta1,theta2,theta3,theta4).dot(np.array([0,0,1]))

  #__________________calculate joint angles_______________________
  def get_theta(self,x_measured):
    a = np.array([0,0,1])
    b = x_measured[2]-x_measured[1]
    b = b/np.linalg.norm(b)
    v = np.cross(a,b)
    s = np.linalg.norm(v)
    c = a.dot(b)
    vx = np.array([[0,-v[2],v[1]],
    		   [v[2],0,-v[0]],
		   [-v[1],v[0],0]])
    one = np.array([[1,0,0],[0,1,0],[0,0,1]])
    mat = one+vx+vx.dot(vx)*(1-c)/s**2

    theta1 = np.arctan2(-mat[0,1],mat[1,1])
    theta2 = np.arctan2(mat[2,1],np.sqrt(mat[2,0]**2+mat[2,2]**2))
    theta3 = np.arctan2(-mat[2,0],mat[2,2])
    theta = np.array([theta1,theta2,theta3])

    d = x_measured[3]-x_measured[2]
    dd = self.roty(theta[2]).dot(self.rotx(-theta[1]).dot(self.rotz(-theta[0]).dot(d)))
    theta4 = np.arctan2(-dd[1],dd[2])
    theta = np.append(theta,theta4)
    return theta,mat
    


  # _______________Recieve data, process it, and publish______________________
  def callback2(self,data):
    #Recieve the image
    try:
      self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    
    self.blob_pos2=Float64MultiArray()
    self.blob_pos2.data=np.array([self.detect_yellow(self.cv_image2),self.detect_blue(self.cv_image2),self.detect_green(self.cv_image2),self.detect_red(self.cv_image2)]).flatten()
    im2=cv2.imshow('window2', self.cv_image2)
    cv2.waitKey(1)

    #Publish the results
    try: 
      self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
      self.blob_pub2.publish(self.blob_pos2)
    except CvBridgeError as e:
      print(e)



  #_________________________combine both images____________________________
  def callbackmaster(self,data):
    #get the blob positions relative to the blue blob (blue at (0,0,0))
    blob_pos1 = np.array(data.data)
    pos_cam1 = self.get_projection(blob_pos1,5/134.)
    pos_cam2 = self.get_projection(self.blob_pos2.data,5/132.)
    x_measured = self.blobs_measured(pos_cam1,pos_cam2)

    #define desired joint angles
    q_d = [ 1.5, 1, -0.5, 0]		#move robot here
    self.joint1=Float64()
    self.joint1.data= q_d[0]
    self.joint2=Float64()
    self.joint2.data= q_d[1]
    self.joint3=Float64()
    self.joint3.data= q_d[2]
    self.joint4=Float64()
    self.joint4.data= q_d[3]
   
    theoretical_green_blob = self.pos_green_blob(q_d[0],q_d[1],q_d[2])
    theoretical_red_blob = self.pos_red_blob(theoretical_green_blob,*q_d)
    theoretical_pos = np.array([0,0,theoretical_green_blob,theoretical_red_blob])
    theta_theoretical, mat_theoretical = self.get_theta(theoretical_pos)

    theta_measured, mat_measured = self.get_theta(x_measured)
    mat_straightforward = self.rot_123(q_d[0],q_d[1],q_d[2])
    eff_sf = mat_straightforward.dot(np.array([0,0,1]))
    eff_th = mat_theoretical.dot(np.array([0,0,1]))
    eff_meas = mat_measured.dot(np.array([0,0,1]))
    print(eff_sf)
    print(eff_th)
    print(eff_meas)
    #print("measured mat:\t{}".format(mat_measured))
    #print("theoretical mat:\t{}".format(mat_theoretical))
    #print("straight mat:\t{}".format(mat_straightforward))
    #print("theo pos:\t{}".format(theoretical_green_blob))
    #print("measured pos:\t{}\n".format(x_measured[2]))
    #print("analytical:\t{}".format(theta_measured))
    #print("desired:\t{}".format(q_d))
    #print("theoretical:\t{}\n\n".format(theta_theoretical))
    
    #publish results
    try: 
      self.robot_joint1_pub.publish(self.joint1)
      self.robot_joint2_pub.publish(self.joint2)
      self.robot_joint3_pub.publish(self.joint3)
      self.robot_joint4_pub.publish(self.joint4)
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


