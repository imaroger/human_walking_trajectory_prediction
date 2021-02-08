#!/usr/bin/python

import os
import rospy
import time
import roslaunch
import subprocess


if __name__ == "__main__":
    roscore = subprocess.Popen('roscore')
    time.sleep(1)

    # Start RViz 
    print("Starting RViz")
    RViz_node = roslaunch.core.Node('rviz', 'rviz',name='rviz')

    RViz_launch=roslaunch.scriptapi.ROSLaunch()
    RViz_launch.start()

    RViz = RViz_launch.launch(RViz_node)
    time.sleep(2)

    # Start estimation
    rospy.init_node('Estimation', anonymous=True)

    pkg_name='estimation'
    executable='marker_rviz.py'
    node_name='RVizMarkers'
    marker_RViz_node = roslaunch.core.Node(pkg_name, executable,name=node_name)

    launch_marker_RViz =roslaunch.scriptapi.ROSLaunch()
    launch_marker_RViz.start()

    launch_marker_RViz.launch(marker_RViz_node)

    print("Enter to set the parameters ...")
    raw_input() 

    ## Choose the human trajectory to consider
    ## human_traj is in ['N1500', 'E1500', 'S1500', 'O1500', 'N4000', 'E4000', 
    ## 'S4000', 'O4000', 'N-0615', 'E-0615', 'S-0615', 'O-0615', 'N0615',
    ## 'E0615', 'S0615', 'O0615', 'N1515', 'E1515', 'S1515', 'O1515', 'N4015',
    ## 'E4015', 'S4015', 'O4015', 'N-0640', 'E-0640', 'S-0640', 'O-0640', 'N0640',
    ## 'E0640', 'S0640', 'O0640', 'N1540', 'E1540', 'S1540', 'O1540', 'N4040',
    ## 'E4040', 'S4040', 'O4040']
    ## subject_id is in [0:9]
    rospy.set_param('human_traj', 'E1540') #E1540, N1515 (subject4)
    rospy.set_param('subject_id', 4)    

    # Choose the parameters for the estimation process
    rospy.set_param('T_0', 50)
    rospy.set_param('rate',8)
    rospy.set_param('N_OC', 75) 
    rospy.set_param('d',0)    

    # Define if the human starts walking (t<T_0), walks (T_0<t<T_f) or stops (t>T_f)
    rospy.set_param('human_status', "Start")   

    print("... done")

    print("Enter to start the estimation process ...")
    raw_input()

    executable='estimation_oc.py'
    node_name='Estimation'
    estimation_node = roslaunch.core.Node(pkg_name, executable,name=node_name)

    launch_estimation =roslaunch.scriptapi.ROSLaunch()
    launch_estimation.start()

    launch_estimation.launch(estimation_node)

    executable='nmpc_online.py'
    node_name='Nmpc'
    nmpc_node = roslaunch.core.Node(pkg_name, executable,name=node_name)

    launch_nmpc =roslaunch.scriptapi.ROSLaunch()
    launch_nmpc.start()

    launch_estimation.launch(nmpc_node)

    executable='human_trajectory.py'
    node_name='HumanTrajectory'
    human_node = roslaunch.core.Node(pkg_name, executable,name=node_name)

    launch_human =roslaunch.scriptapi.ROSLaunch()
    launch_human.start()

    launch_human.launch(human_node)


    print("... done")

    # r = rospy.Rate(1)
    while not rospy.is_shutdown():
        print("Running")
        if not RViz.is_alive():
            print("Stopping RViz")
            rospy.signal_shutdown("RViz stop running")
            # Terminate the roscore subprocess
            print("Stop roscore")
            roscore.terminate()
        rospy.spin()

