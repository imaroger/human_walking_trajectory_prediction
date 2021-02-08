#!/usr/bin/python

import os, sys
import time
import numpy as np
np.set_printoptions(threshold=np.nan, linewidth =np.nan)
from walking_generator.visualization_traj import PlotterTraj
from walking_generator.combinedqp_traj import NMPCGeneratorTraj
from walking_generator.interpolation_traj import Interpolation

from math import sqrt,floor
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

import rospy
from estimation.msg import TrajMsg, NmpcMsg
from math import cos,sin,pi,sqrt
from std_msgs.msg import Bool, Float64

def resizeTraj(x, y, theta, velocity_ref):
    traj_length = len(x)
    # print("lenx",traj_length)

    okay = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) > 0)
    x,y = x[okay],y[okay]
    # print(x,y)
    tck, u = splprep([x, y], s=0)
    unew = np.linspace(0,1,traj_length)
    data = splev(unew, tck)
    x,y = data[0],data[1]

    ind = np.where(np.abs(np.diff(theta))>0.2)

    max_delta_ori = np.max(np.abs(np.diff(theta)))
    if max_delta_ori < 0.8:
        velocity_low = 0.05
    elif max_delta_ori < 1.7:
        velocity_low = 0.001
    elif max_delta_ori < 2.8:
        velocity_low = 0.0005
    else:
        velocity_low = 0.0001

    # print("vel_low",velocity_low)

    ind_partition, d  = [[0]], []
    i,previous = 0,"ref"
    while i < traj_length-1:
        if np.sum(np.isin(ind,i)) == 0:
            if previous == "low":     
                ind_partition.append([])           
                ind_partition[-1].append(i)
            ind_partition[-1].append(i+1)
            previous = "ref"
            i+=1
        else:
            if previous == "ref":
                ind_partition.append([])              
                ind_partition[-1].append(i)
            ind_partition[-1].append(i+1)                           
            previous = "low"
            i+=1

    # print("ind_part",ind_partition)

    new_length_list = []

    for k in range(len(ind_partition)):
        d.append(0)
        for i in ind_partition[k][:-1]:
            d[-1] += sqrt((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2)
        if k%2 == 0:
            t = (d[-1])/velocity_ref
        else:
            t = d[-1]/velocity_low
        # print(t)
        new_length_list.append(int((floor(t/0.2))))   
    # print("d",d[-1]/velocity_ref,(d[-1]+delta_d)/velocity_ref)
    # print("new_len",new_length_list)     
    
    new_x,new_y,new_theta = np.array([]),np.array([]),np.array([])
    i = 0

    if np.sum(new_length_list) > 16:
        for length in new_length_list:
            if length != 0:
                ind = np.array(ind_partition[i])
                current_x,current_y,current_theta = x[ind],y[ind],theta[ind]

                new_time = np.linspace(0,1,length)
                old_time = np.linspace(0,1,len(ind))
                current_x = np.interp(new_time,old_time,current_x) 
                current_y = np.interp(new_time,old_time,current_y)              
                current_theta = np.interp(new_time,old_time,current_theta)  

                new_x = np.concatenate((new_x,current_x))
                new_y = np.concatenate((new_y,current_y))
                new_theta = np.concatenate((new_theta,current_theta))
            i += 1
    else:
        new_time = np.linspace(0,1,16)
        old_time = np.linspace(0,1,len(x))
        new_x = np.interp(new_time,old_time,x) 
        new_y = np.interp(new_time,old_time,y)              
        new_theta = np.interp(new_time,old_time,theta)  

    new_traj = np.zeros((3,len(new_x)), dtype=float)
    new_traj[0],new_traj[1],new_traj[2] = new_x,new_y,new_theta
    return new_traj

def initToZero(x_0, y_0, theta_0, x, y, theta):
    x_local,y_local = (x-x_0)*np.cos(theta_0) + (y-y_0)\
        *np.sin(theta_0), -(x-x_0)*np.sin(theta_0) + \
        (y-y_0)*np.cos(theta_0)
    theta_local = theta - theta_0 
    return x_local, y_local, theta_local

def zeroToInit(x_0, y_0, theta_0, x_local, y_local, theta_local):
    x,y = x_0 + x_local*cos(theta_0) - y_local*sin(theta_0),\
        y_0 + x_local*sin(theta_0) + y_local*cos(theta_0)
    theta = theta_0+theta_local
    return x,y,theta

def nmpcResults2Msg(comx,comy,comz,comq,footx,footy,footq,foot,future_footx,\
        future_footy,future_footq):
    msg = NmpcMsg()

    msg.com_pose.position.x = comx
    msg.com_pose.position.y = comy
    msg.com_pose.position.z = comz

    msg.com_pose.orientation.x = 0
    msg.com_pose.orientation.y = 0
    msg.com_pose.orientation.z = sin(comq/2)     
    msg.com_pose.orientation.w = cos(comq/2) 

    msg.foot_pose.position.x = footx
    msg.foot_pose.position.y = footy
    msg.foot_pose.position.z = 0   

    msg.foot_pose.orientation.x = 0
    msg.foot_pose.orientation.y = 0
    msg.foot_pose.orientation.z = sin(footq/2)      
    msg.foot_pose.orientation.w = cos(footq/2)  

    msg.future_foot_pose.position.x = future_footx
    msg.future_foot_pose.position.y = future_footy
    msg.future_foot_pose.position.z = 0   

    msg.future_foot_pose.orientation.x = 0
    msg.future_foot_pose.orientation.y = 0
    msg.future_foot_pose.orientation.z = sin(future_footq/2)      
    msg.future_foot_pose.orientation.w = cos(future_footq/2)  

    msg.foot = foot

    return msg

def findGoodInd(x,y,N_OC,T_0,d):
    start = []
    if d < 0:
        for i in range(T_0,-1,-1):
            d_i = sqrt((x[T_0]-x[i])**2+(y[T_0]-y[i])**2)
            if d_i >= abs(d) :
                start.append(i)
        if len(start) == 0:
            return 0
        else:
            return start[0]
    else :
        ind_f = T_0+int((N_OC-T_0)/2)
        for i in range(T_0,ind_f+1,1):
            d_i = sqrt((x[T_0]-x[i])**2+(y[T_0]-y[i])**2)
            if d_i >= d:
                start.append(i)
        if len(start) == 0:
            return ind_f
        else:
            return start[0]      

########################################################################
################################## MAIN ################################
########################################################################

class estimation_pub:

    def __init__(self):
        self.sub = rospy.Subscriber("estimated_trajectory", TrajMsg, self.callback)
        self.pub = rospy.Publisher("nmpc_generator", NmpcMsg, queue_size=10)
        self.pub_qp_solver_cv = rospy.Publisher("qp_solver_cv", Bool, queue_size=10)
        self.pub_vel = rospy.Publisher("human_vel",Float64, queue_size=10)
        self.pub_dist = rospy.Publisher("dist_human_robot",Float64, queue_size=10)
        self.r = rospy.get_param('rate')

        # instantiate pattern generator
        self.nmpc = NMPCGeneratorTraj(fsm_state='L/R')
        self.nmpc.set_security_margin(0.09, 0.05)

        # set initial values 
        comx = [0.00679821, 0.0, 0.0]
        comy = [0.08693283,0.0, 0.0]
        comz = 8.92675352e-01
        footx = 0.00949035
        footy = 0.095
        footq = 0.0
        # self.foot='left'
        # self.comq = [0.0,0.0, 0.0]
        self.nmpc.set_initial_values(comx, comy, comz, \
            footx, footy, footq, 'left')

        self.interp_nmpc = Interpolation(0.001,self.nmpc)



        self.x0,self.y0,self.th0 = 0,0,0

        self.d = rospy.get_param('d')
        self.T_0 = rospy.get_param('T_0')
        self.N_OC = rospy.get_param('N_OC')
        self.ind_start = 0
        self.ind_current = 0
        self.count = 0

    def callback(self, traj):
        done = True
        d = self.d
        T_0 = self.T_0
        N_OC = self.N_OC
        status = rospy.get_param('human_status')
        x, y, theta = traj.x_traj, traj.y_traj, traj.theta_traj
        # print("---",status, len(x))

        if len(x) > 1:
            print(x[T_0],y[T_0])
            dist_from_start = sqrt((x[T_0]-x[0])**2+(y[T_0]-y[0])**2)
            dist_dpos = sqrt((x[T_0]-x[T_0+int((N_OC-T_0)/2)])**2+(y[T_0]-y[T_0+int((N_OC-T_0)/2)])**2)

            print("dist_avt : ",d,dist_from_start,dist_dpos)

            if (d == 0 or (d < 0 and abs(d) <= dist_from_start) or (d > 0 and d <= dist_dpos)):
                if self.x0 == 0 and self.y0 == 0 and self.th0 == 0:
                    if d == 0:
                        self.ind_start = T_0
                        self.ind_current = T_0
                    else :
                        ind = findGoodInd(x,y,N_OC,T_0,d)
                        self.ind_start,self.ind_current = ind,ind

                    self.x0,self.y0,self.th0 = x[self.ind_start],y[self.ind_start],theta[self.ind_start] 
                
                else :
                    if d != 0:
                        self.ind_current = findGoodInd(x,y,N_OC,T_0,d)             

                # x0, y0, th0 = x[self.ind_start],y[self.ind_start],theta[self.ind_start]     
                velocity_ref = (sqrt((x[T_0]-x[0])**2+(y[T_0]-y[0])**2))*self.r/(T_0+1)
                # print("vel : ",sqrt((x[T_0]-x[0])**2+(y[T_0]-y[0])**2)*self.r/(T_0+1))
                vel_msg = Float64()
                vel_msg.data = (sqrt((x[T_0]-x[0])**2+(y[T_0]-y[0])**2))*self.r/(T_0+1)
                self.pub_vel.publish(vel_msg) 

                x, y, theta = initToZero(self.x0, self.y0, self.th0, \
                    np.array(x[self.ind_current:]), np.array(y[self.ind_current:]), np.array(theta[self.ind_current:]))

                resized_traj = resizeTraj(x, y, theta, velocity_ref)



                trajectory_reference = resized_traj[:,0:16]

                self.nmpc.   set_trajectory_reference(trajectory_reference)

                # solve QP
                nb_failures = self.nmpc.   solve()
                self.nmpc.   simulate()
                self.interp_nmpc.interpolate(self.count*0.2)

                # initial value embedding by internal states and simulation
                comx, comy, comz, footx, footy, footq, foot, comq, future_footx,\
                    future_footy, future_footq = \
                self.nmpc.update()
                self.nmpc.set_initial_values(comx, comy, comz, \
                    footx, footy, footq, foot, comq)

                if nb_failures != 0:
                    done = False      

                # print(done)
                self.pub_qp_solver_cv.publish(done)

                if nb_failures <= 2:

                    # self.comx = comx
                    # self.comy = comy
                    # self.comz = comz
                    # self.comq = comq
                    # self.footx = footx
                    # self.footy = footy
                    # self.footq = footq
                    # self.foot = foot
                    
                    comx, comy, comq = zeroToInit(self.x0, self.y0, self.th0, comx[0], comy[0], comq[0])
                    footx, footy, footq = zeroToInit(self.x0, self.y0, self.th0, footx, footy, footq)
                    future_footx, future_footy, future_footq = zeroToInit(self.x0, self.y0, self.th0, future_footx, future_footy, future_footq)     

                    dist = sqrt((traj.x_traj[T_0]-comx)**2+(traj.y_traj[T_0]-comy)**2)

                    # if d > 0:
                    #     delta_d = d - dist
                    # elif d < 0:
                    #     delta_d = d + dist                                          

                    # print("dist : ",dist)  
                    dist_msg = Float64()
                    dist_msg.data = dist    
                    self.pub_dist.publish(dist_msg)       

                    nmpc_msg = nmpcResults2Msg(comx,comy,comz,comq,footx,footy,footq,\
                        foot,future_footx, future_footy, future_footq)
                    self.pub.publish(nmpc_msg)
                    self.count += 1
                else:
                    print("*** QP failed ***")
                    nmpc_msg = nmpcResults2Msg(0,0,0,0,0,0,0,"none",0,0,0)
                    self.pub.publish(nmpc_msg)  

    def save_data(self):
        self.interp_nmpc.save_to_file("/local/imaroger/catkin_ws/src/estimation/src/data/nmpc_traj_online.csv")
        

if __name__ == '__main__':
    try:
        rospy.init_node('NmpcOnline', anonymous=True)
        estimator = estimation_pub()
        
        while not rospy.is_shutdown():
            rospy.spin()
            estimator.save_data()
    except rospy.ROSInterruptException:
        print("EstimationOC Shutting down")

