#!/usr/bin/env python

import rospy
from visualization_msgs.msg import Marker
from estimation.msg import TrajMsg, NmpcMsg
import numpy as np
from math import cos,sin,pi
import message_filters
# import tf.transformations as tt

def setMarker(marker,pos,q,scale,color):
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

def declareMarker(marker_ns,marker_id,path):
	marker = Marker()
	marker.header.frame_id = "/world" 
	marker.header.stamp = rospy.Time.now()

	# Set the namespace and id for this marker.  This serves to create a unique ID
	# Any marker sent with the same namespace and id will overwrite the old one
	marker.ns = marker_ns
	marker.id = marker_id 

	# Set the marker type. 
	if path != "none":
		marker.type = marker.MESH_RESOURCE
		marker.mesh_resource = path
	else:
		marker.type = marker.SPHERE

	# Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
	marker.action = marker.ADD 

	return marker

def addTable():
	# Send the initial position of the table 
	marker = declareMarker("Table", 0, "package://human_walking_trajectory_prediction/mesh/Table.stl")
	marker = setMarker(marker,[0,1.2,0.75],[0,0,0.7,0.7],1,[1,1,1,1])
	marker.lifetime = rospy.Duration()
	return marker

def addHuman(x,y,theta,i):
	# Send the current position of the human mesh
	mesh = declareMarker("Human", 0, "package://human_walking_trajectory_prediction/mesh/Human.stl")

	th = theta+pi/2
	current_pos = [-0.09*sin(th)+x,0.09*cos(th)+y,0.08]
	current_q = [0,0,sin(th/2),cos(th/2)]

	mesh = setMarker(mesh,current_pos,current_q,0.001,[1,1,1,0.5])
	mesh.lifetime = rospy.Duration(5)

	return mesh

def addMarker(x,y,z,name,color,i):
	# Send the current position of the CoM and record it
	marker = declareMarker(name, i, "none")
	marker = setMarker(marker,[x,y,z],[0,0,0,1],0.02,color)
	marker.lifetime = rospy.Duration()
	return marker

def addCurrentTraj(x,y,i):
	# Display the trajectory of the human CoM which is sent to the estimation process
	marker = declareMarker("CoM", i, "none")
	marker = setMarker(marker,[x,y,0.94],[0,0,0,1],0.04,[1,1,0,1])
	marker.lifetime = rospy.Duration(5)
	return marker

def addCurrentEst(x,y,i):
	# Display the trajectory of the human CoM which is sent to the estimation process
	marker = declareMarker("CoM_est", i, "none")
	marker = setMarker(marker,[x,y,0.9],[0,0,0,1],0.04,[1,0,1,1])
	marker.lifetime = rospy.Duration(5)
	return marker

def addSupportFoot(x,y,q_z,q_w,name,i):
	if name == "Old_support_foot":
		ind, color, z = i, [0.8,0.8,0.8,0.8], 0
	elif name == "Current_support_foot":
		ind, color, z = 0, [1,0,0,1], 0.001
	else:
		ind, color, z = 0, [0,1,0,1], 0.001
	mesh = declareMarker(name, ind, "package://human_walking_trajectory_prediction/mesh/RobotFoot.stl")
	current_q = [0,0,q_z,q_w]
	mesh.lifetime = rospy.Duration()
	mesh = setMarker(mesh,[x,y,z],current_q,0.001,color)	
	return mesh	

class marker_pub:

	def __init__(self):
		human_sub = message_filters.Subscriber('human_trajectory', TrajMsg)
		estimation_sub = message_filters.Subscriber('estimated_trajectory', TrajMsg)
		nmpc_sub = message_filters.Subscriber('nmpc_generator', NmpcMsg)

		ts = message_filters.ApproximateTimeSynchronizer([human_sub, estimation_sub, nmpc_sub],\
			100, 0.1, allow_headerless=True)
		ts.registerCallback(self.callback)

		self.pub = rospy.Publisher("visualization_marker", Marker, queue_size=10)
		self.count = 0

		self.old_foot = "left"
		self.old_pose_foot = [0,0,0,0]

	def callback(self, traj, est, nmpc):
		# Display the motionless table
		mesh_table = addTable()
		self.pub.publish(mesh_table)

		x, y, theta = traj.x_traj, traj.y_traj, traj.theta_traj
		x_est, y_est, theta_est = est.x_traj, est.y_traj, est.theta_traj

		com_x, com_y, com_z = nmpc.com_pose.position.x,nmpc.com_pose.position.y,nmpc.com_pose.position.z
		foot_x, foot_y = nmpc.foot_pose.position.x,nmpc.foot_pose.position.y
		footq_z, footq_w = nmpc.foot_pose.orientation.z,nmpc.foot_pose.orientation.w
		future_foot_x, future_foot_y = nmpc.future_foot_pose.position.x,nmpc.future_foot_pose.position.y
		future_footq_z, future_footq_w = nmpc.future_foot_pose.orientation.z,nmpc.future_foot_pose.orientation.w
		
		foot = nmpc.foot

		# Display current human mesh and its CoM trajectory (green)
		mesh_human = addHuman(x[-1],y[-1],theta[-1],self.count)
		# marker_human = addMarker(x[-1],y[-1],0.9,"Human_com",[0,1,0,1],self.count)
		self.pub.publish(mesh_human)
		# self.pub.publish(marker_human)

		# Display the current human trajectory sent to the estimation process (pink)
		for i in range(len(x)):
			marker_traj = addCurrentTraj(x[i], y[i], i)
			self.pub.publish(marker_traj)

		# Display the current estimated trajectory (yellow)			
		for i in range(len(x_est)):
			marker_est = addCurrentEst(x_est[i], y_est[i], i)
			self.pub.publish(marker_est)

		# Display the current (red) and old (grey) support foot of the robot	
		if foot != self.old_foot:
			mesh_old_foot = addSupportFoot(self.old_pose_foot[0],\
				self.old_pose_foot[1], self.old_pose_foot[2], self.old_pose_foot[3],\
				"Old_support_foot", self.count)
			self.pub.publish(mesh_old_foot)

		mesh_current_foot = addSupportFoot(foot_x, foot_y, footq_z, footq_w,\
			"Current_support_foot", self.count)
		self.pub.publish(mesh_current_foot)

		mesh_future_foot = addSupportFoot(future_foot_x, future_foot_y, future_footq_z, future_footq_w,\
			"Future_support_foot", self.count)
		self.pub.publish(mesh_future_foot)		

		# Display the CoM trajectory of the robot (red)
		marker_robot = addMarker(com_x,com_y,com_z,"Robot_com",[1,0,0,1],self.count)
		self.pub.publish(marker_robot)

		self.old_pose_foot = [foot_x,foot_y,footq_z,footq_w]
		self.count += 1

		

if __name__ == '__main__':
	try:
		rospy.init_node('RVizMarkers', anonymous=True)
		marker_pub()
		while not rospy.is_shutdown():
			rospy.spin()
	except rospy.ROSInterruptException:
	    print("RVizMarkers Shutting down")
