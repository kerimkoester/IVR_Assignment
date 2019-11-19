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
from scipy.optimize import minimize
import math


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
    cv2.waitKey(1)
    self.image1_sub = rospy.Subscriber("/camera1/blob_pos",Float64MultiArray,self.callbackmaster)
    # initialize a publisher to publish position of blobs
    self.blob_pub2 = rospy.Publisher("/camera2/blob_pos",Float64MultiArray, queue_size=10)
    # initialize a publisher to send joints' angular position to the robot
    self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
    self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
    self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
    self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
    # initialize publisher to publish measured joint angles
    self.robot_joint1_measured = rospy.Publisher("/robot/joint1_position_measured", Float64, queue_size=10)
    self.robot_joint2_measured = rospy.Publisher("/robot/joint2_position_measured", Float64, queue_size=10)
    self.robot_joint3_measured = rospy.Publisher("/robot/joint3_position_measured", Float64, queue_size=10)
    self.robot_joint4_measured = rospy.Publisher("/robot/joint4_position_measured", Float64, queue_size=10)
    # initalize a publisher to publish measured target sphere
    self.target_xpos = rospy.Publisher("/target/x_position_measured", Float64, queue_size=10)
    self.target_ypos = rospy.Publisher("/target/y_position_measured", Float64, queue_size=10)
    self.target_zpos = rospy.Publisher("/target/z_position_measured", Float64, queue_size=10)
    # initalize a publisher to publish measured end-effector
    self.end_effector_xpos = rospy.Publisher("/end_effector/x_position_measured", Float64, queue_size=10)
    self.end_effector_ypos = rospy.Publisher("/end_effector/y_position_measured", Float64, queue_size=10)
    self.end_effector_zpos = rospy.Publisher("/end_effector/z_position_measured", Float64, queue_size=10)
    # initialize reserve variables for the case of an error
    self.pos_reserve = np.array([0,0,0,0,0,0,0,0,0,0])
    self.spherex_reserve=0
    self.spherey_reserve=0
    self.spherez_reserve=0
    # record the beginning time
    self.time_trajectory = rospy.get_time()
    # initialize errors
    self.time_previous_step = np.array([rospy.get_time()], dtype='float64')     
    # initialize error and derivative of error for trajectory tracking  
    self.error = np.array([0.0,0.0,0.0], dtype='float64')  
    self.error_d = np.array([0.0,0.0,0.0], dtype='float64')
    self.q_d = np.array([0.0,0.0,0.0,0.0])

    

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

  def get_projection(self,blobs,target,scale):
    corrected_blobs = self.eliminate_nonvisible_blobs(blobs)
    pos_cam = np.array(corrected_blobs).reshape((4,2))
    pos_cam = scale*pos_cam
    pos_cam_target = scale*target
    pos_cam_target = pos_cam_target-pos_cam[1]
    pos_cam = pos_cam-pos_cam[1]
    pos_cam[:,1]=-pos_cam[:,1]
    pos_cam_target[1] = -pos_cam_target[1]
    return np.array(pos_cam),pos_cam_target


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

  def target_measure(self,target1,target2):
    return np.array([target2[0],target1[0],self.z_average(target1[1],target2[1],target2[0],target1[0])])
    

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

  def pos_red_blob_ana(self,theta1,theta2,theta3,theta4):
    s1 = np.sin(theta1)
    c1 = np.cos(theta1)
    s2 = np.sin(theta2)
    c2 = np.cos(theta2)
    s3 = np.sin(theta3)
    c3 = np.cos(theta3)
    s4 = np.sin(theta4)
    c4 = np.cos(theta4)
    f = 3+2*c4
    x = 2*s1*c2*s4+f*(c1*s3+s1*s2*c3)
    y = -2*c1*c2*s4+f*(s1*s3-c1*s2*c3)
    z = -s2*s4+f*c2*c3
    return np.array([x,y,z])

  #__________________calculate joint angles_______________________
 
  #function that is minimized w.r.t. theta_123
  def func_min(self,theta):
    s1 = np.sin(theta[0])
    c1 = np.cos(theta[0])
    s2 = np.sin(theta[1])
    c2 = np.cos(theta[1])
    s3 = np.sin(theta[2])
    c3 = np.cos(theta[2])
    #define unity vector pointing from blue blob to green blob
    b = self.x_measured[2]-self.x_measured[1]
    b = b/np.linalg.norm(b)
    return abs(c1*s3+s1*s2*c3-b[0])+abs(s1*s3-c1*s2*c3-b[1])+abs(c2*c3-b[2])
  def measure_angle(self):
    theta_num = minimize(self.func_min,self.q_d[:-1],method='nelder-mead',options={'xtol':1e-6})
    #theta_num = minimize(self.func_min,np.array([0,0,0]),method='nelder-mead',options={'xtol':1e-8})
    theta_num = theta_num.x
    d = self.x_measured[3]-self.x_measured[2]
    d = d/np.linalg.norm(d)
    dd = self.roty(-theta_num[2]).dot(self.rotx(-theta_num[1]).dot(self.rotz(-theta_num[0]).dot(d)))
    theta4 = np.arctan2(-dd[1],dd[2])
    theta_num = np.append(theta_num,theta4)
    return theta_num

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

  #__________________________robot control_____________________________________
  def Jacobian(self,theta):
    s1 = np.sin(theta[0])
    c1 = np.sin(theta[0])
    s2 = np.sin(theta[1])
    c2 = np.sin(theta[1])
    s3 = np.sin(theta[2])
    c3 = np.sin(theta[2])
    s4 = np.sin(theta[3])
    c4 = np.sin(theta[3])
    J = np.array([[2*c1*c2*s4+(3+2*c4)*(-s1*s3+c1*s2*c3),-2*s1*s2*s4+(3+2*c4)*s1*c2*c3,(3+2*c4)*(c1*c3-s1*s2*s3),2*s1*c2*c4-2*s4*(c1*s3+s1*s2*c3)],
  	     	  [2*s1*c2*s4+(3+2*c4)*(c1*s3+s1*s2*c3),2*c1*s2*s4-(3+2*c4)*c1*c2*c3,(3+2*c4)*(s1*c3+c1*s2*s3),-2*c1*c2*c4-2*s4*(s1*s3-c1*s2*c3)],
		  [0,-c2*s4-(3+2*c4)*s2*c3,-(3+2*c4)*c2*s3,-s2*c4-2*s4*c2*c3]])
    return J

  def control_closed(self,pos_d,pos,q):
    # P gain
    K_p = np.array([[0.0001,0,0],[0,0.0001,0],[0,0,0.0001]])
    # D gain
    K_d = np.array([[0,0,0],[0,0.0,0],[0,0,0]])
    # estimate time step
    cur_time = np.array([rospy.get_time()])
    dt = cur_time - self.time_previous_step
    self.time_previous_step = cur_time
    # estimate derivative of error
    self.error_d = ((pos_d - pos) - self.error)/dt
    # estimate error
    self.error = pos_d-pos
    J_inv = np.linalg.pinv(self.Jacobian(q))  # calculating the psudeo inverse of Jacobian
    dq_d =np.dot(J_inv, ( np.dot(K_d,self.error_d.transpose()) + np.dot(K_p,self.error.transpose()) ) )  # control input (angular velocity of joints)
    print(dq_d)
    q_d = q + (dt * dq_d)  # control input (angular position of joints)
    return q_d





  # _______________Recieve data, process it, and publish______________________
  def callback2(self,data):
    #Recieve the image
    try:
      self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    
    self.orange_sphere2=self.detectTarget(self.cv_image2)
    self.blob_pos2=Float64MultiArray()
    self.blob_pos2.data=np.array([self.detect_yellow(self.cv_image2),self.detect_blue(self.cv_image2),self.detect_green(self.cv_image2),self.detect_red(self.cv_image2),self.orange_sphere2]).flatten()
    im2=cv2.imshow('window2', self.cv_image2)
    cv2.waitKey(1)
    #cv2.imwrite('cam2.png',self.cv_image2)

    #Publish the results
    try: 
      self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
      self.blob_pub2.publish(self.blob_pos2)
    except CvBridgeError as e:
      print(e)
  


  #_________________________combine both images____________________________
  def callbackmaster(self,data):
    #get the blob positions relative to the blue blob (blue at (0,0,0))
    if len(self.blob_pos2.data)==0:
      pos2 = self.pos_reserve
    else:
      pos2 = self.blob_pos2.data
      self.pos_reserve = pos2
    blob_pos1 = np.array(data.data[0:8])
    blob_pos2 = np.array(pos2[0:8])
    target_pos1 = np.array(data.data[8:])
    target_pos2 = pos2[8:]
      
    pos_cam1,target_cam1 = self.get_projection(blob_pos1,target_pos1,5/134.)
    pos_cam2,target_cam2 = self.get_projection(blob_pos2,target_pos2,5/132.)
    self.x_measured = self.blobs_measured(pos_cam1,pos_cam2)
    self.target_measured = self.target_measure(target_cam1,target_cam2)

   
    #prepare publishing measured joint angles
    theta_measured = self.measure_angle()
    self.joint1m=Float64()
    self.joint1m.data= theta_measured[0]
    self.joint2m=Float64()
    self.joint2m.data= theta_measured[1]
    self.joint3m=Float64()
    self.joint3m.data= theta_measured[2]
    self.joint4m=Float64()
    self.joint4m.data= theta_measured[3]

    #prepare publishing measured target position
    self.spherex=Float64()
    self.spherex.data=self.target_measured[0]
    self.spherey=Float64()
    self.spherey.data=self.target_measured[1]
    self.spherez=Float64()
    self.spherez.data=self.target_measured[2]
    #Correct error when sphere is hidden by target
    #Assume that the first value of sphere isn't less than -1
    if self.spherex.data<-1:
       self.spherex.data = self.spherex_reserve
    else:
       self.spherex_reserve = self.spherex.data

    if self.spherey.data <-1:
       self.spherey.data = self.spherey_reserve
    else:
       self.spherey_reserve = self.spherey.data
 
    if self.spherez.data <-1:
       self.spherez.data = self.spherez_reserve
    else:
       self.spherez_reserve = self.spherez.data

    pos_d = np.array([self.spherex.data,self.spherey.data,self.spherez.data])
    pos = self.x_measured[3]

    self.q_d = self.control_closed(pos_d,pos,theta_measured)
    print(self.q_d)

    #define desired joint angles
    #self.q_d = [0.5,0.5,0.5,0.5]		#move robot here
    
    #prepare publishing desired joint angles
    self.joint1=Float64()
    self.joint1.data= self.q_d[0]
    self.joint2=Float64()
    self.joint2.data= self.q_d[1]
    self.joint3=Float64()
    self.joint3.data= self.q_d[2]
    self.joint4=Float64()
    self.joint4.data= self.q_d[3]

    #prepare publishing end-effector position
    self.end_effector_x=Float64()
    self.end_effector_x.data = pos[0]
    self.end_effector_y=Float64()
    self.end_effector_y.data = pos[1]
    self.end_effector_z=Float64()
    self.end_effector_z.data = pos[2]


    #several results for testing
    '''print("desired:\t{}".format(self.q_d))
    print("numerical:\t{}".format(theta_measured))
    print(self.func_min(self.q_d[:-1]))
    print(self.func_min(theta_measured[:-1]))
    print(self.pos_red_blob_ana(*self.q_d))
    print(self.pos_red_blob(self.pos_green_blob(*self.q_d[0:-1]),*self.q_d))'''

    #publish results
    try: 
      self.robot_joint1_pub.publish(self.joint1)
      self.robot_joint2_pub.publish(self.joint2)
      self.robot_joint3_pub.publish(self.joint3)
      self.robot_joint4_pub.publish(self.joint4)
      self.robot_joint1_measured.publish(self.joint1m)
      self.robot_joint2_measured.publish(self.joint2m)
      self.robot_joint3_measured.publish(self.joint3m)
      self.robot_joint4_measured.publish(self.joint4m)
      self.target_xpos.publish(self.spherex)
      self.target_ypos.publish(self.spherey)
      self.target_zpos.publish(self.spherez)
      self.end_effector_xpos.publish(self.end_effector_x)
      self.end_effector_ypos.publish(self.end_effector_y)
      self.end_effector_zpos.publish(self.end_effector_z)
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


