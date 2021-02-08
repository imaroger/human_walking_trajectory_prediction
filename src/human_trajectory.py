#!/usr/bin/env python
import rospy
import numpy as np
from estimation.msg import TrajMsg
from visualization_msgs.msg import Marker

def chooseTraj():
	# Load the choosen human trajectory according to the parameters (name and subject)
	name = rospy.get_param('human_traj')
	subject = rospy.get_param('subject_id')
	path = 'data/Human/' + name + '.dat'
	human_traj = np.loadtxt(path)
	return human_traj[6*subject],human_traj[6*subject+1],human_traj[6*subject+5]

def setMarker(marker_ns,marker_id,pos,q,scale,color):
	marker = Marker()
	marker.header.frame_id = "/world" 
	marker.header.stamp = rospy.Time.now()

	# Set the namespace and id for this marker.  This serves to create a unique ID
	# Any marker sent with the same namespace and id will overwrite the old one
	marker.ns = marker_ns
	marker.id = marker_id 

	# Set the marker type. 
	marker.type = marker.SPHERE

	# Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
	marker.action = marker.ADD 

	# Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
	marker.pose.position.x = pos[0] 
	marker.pose.position.y = pos[1]  
	marker.pose.position.z = pos[2]  
	marker.pose.orientation.x = q[0] 
	marker.pose.orientation.y = q[1] 
	marker.pose.orientation.z = q[2] 
	marker.pose.orientation.w = q[3] 

	# Set the scale of the marker -- 1x1x1 here means 1m on a side
	marker.scale.x = scale 
	marker.scale.y = scale 
	marker.scale.z = scale 

	# Set the color -- be sure to set alpha to something non-zero!
	marker.color.r = color[0] 
	marker.color.g = color[1]
	marker.color.b = color[2] 
	marker.color.a = color[3] 

	return marker

def addTraj(x,y,i):
	# Display the whole trajectory of the human CoM
	marker = setMarker("HumanTraj", i,[x,y,0.94],[0,0,0,1],0.02,[0,1,0,1])
	marker.lifetime = rospy.Duration(5)
	return marker

def sendTraj():
	# Send the travelled trajectory on the topic /human_trajectory
	x,y,theta = chooseTraj()

	i = 0
	T_0 = rospy.get_param('T_0')
	r = rospy.get_param('rate')

	pub = rospy.Publisher("human_trajectory", TrajMsg, queue_size=10)
	traj = TrajMsg()
	rate = rospy.Rate(r) # Hz	




	while not rospy.is_shutdown() and i <= len(x):
		# print(i)

		pub_traj = rospy.Publisher("visualization_marker", Marker, queue_size=10)
		for k in range(len(x)):
			marker_traj = addTraj(x[k], y[k], k)
			pub_traj.publish(marker_traj)

		if i <= T_0-1:
			print("Start",i)
			traj.x_traj, traj.y_traj, traj.theta_traj =\
			x[0:(i+1)],y[0:(i+1)],theta[0:(i+1)]
			if i == T_0-1:
				rospy.set_param('human_status', "Walk")	
		else:
			print("Walk",i)
			traj.x_traj, traj.y_traj, traj.theta_traj =\
			x[i-T_0+1:(i+1)],y[i-T_0+1:(i+1)],theta[i-T_0+1:(i+1)]
		pub.publish(traj)
		# print(len(traj.x_traj),"--- x ---",traj.x_traj)		
		# print(len(traj.y_traj),"--- y ---",traj.y_traj)	
		i += 1
		rate.sleep()
	rospy.set_param('human_status', "Stop")


	while not rospy.is_shutdown():
		print("Stop")
		traj.x_traj, traj.y_traj, traj.theta_traj = [x[-1]],[y[-1]],theta[[-1]]
		pub.publish(traj)
		rate.sleep()


if __name__ == '__main__': 
    rospy.init_node('HumanTrajectory', anonymous = True)
    sendTraj() 
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("HumanTrajectory Shutting down")
