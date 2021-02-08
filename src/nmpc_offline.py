import os, sys
import time
import numpy 
numpy.set_printoptions(threshold=numpy.nan, linewidth =numpy.nan)
from walking_generator.visualization_traj import PlotterTraj
from walking_generator.combinedqp_traj import NMPCGeneratorTraj
from walking_generator.interpolation_traj import Interpolation

from math import sqrt,floor
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42



def resizeTraj(traj,velocity_ref):
    traj_length = len(traj[0])
    x,y,theta = traj[0],traj[1],traj[2]

    okay = numpy.where(numpy.abs(numpy.diff(x)) + numpy.abs(numpy.diff(y)) > 0)
    x,y = x[okay],y[okay]
    tck, u = splprep([x, y], s=0)
    unew = numpy.linspace(0,1,traj_length)
    data = splev(unew, tck)
    x,y = data[0],data[1]

    # ind=[]

    ind = numpy.where(numpy.abs(numpy.diff(theta))>0.2)

    max_delta_ori = numpy.max(numpy.abs(numpy.diff(theta)))
    if max_delta_ori < 0.8:
        velocity_low = 0.05
    elif max_delta_ori < 1.7:
        velocity_low = 0.001
    elif max_delta_ori < 2.8:
        velocity_low = 0.0005
    else:
        velocity_low = 0.0001

    ind_partition, d  = [[0]], []
    i,previous = 0,"ref"
    while i < traj_length-1:
        if numpy.sum(numpy.isin(ind,i)) == 0:
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

    new_length_list = []

    for k in range(len(ind_partition)):
        d.append(0)
        for i in ind_partition[k][:-1]:
            d[-1] += sqrt((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2)
        if k%2 == 0:
            t = d[-1]/velocity_ref
        else:
            t = d[-1]/velocity_low
        new_length_list.append(int((floor(t/0.1))))        

    new_x,new_y,new_theta = numpy.array([]),numpy.array([]),numpy.array([])
    i = 0
    for length in new_length_list:
        if length != 0:
            ind = numpy.array(ind_partition[i])
            current_x,current_y,current_theta = x[ind],y[ind],theta[ind]

            new_time = numpy.linspace(0,1,length)
            old_time = numpy.linspace(0,1,len(ind))
            current_x = numpy.interp(new_time,old_time,current_x) 
            current_y = numpy.interp(new_time,old_time,current_y)              
            current_theta = numpy.interp(new_time,old_time,current_theta)  

            new_x = numpy.concatenate((new_x,current_x))
            new_y = numpy.concatenate((new_y,current_y))
            new_theta = numpy.concatenate((new_theta,current_theta))
        i += 1

    new_traj = numpy.zeros((3,len(new_x)), dtype=float)
    new_traj[0],new_traj[1],new_traj[2] = new_x,new_y,new_theta
    return new_traj

def translate(traj):
    x_0,y_0,theta_0 = traj[0][0],traj[1][0],traj[2][0]
    traj[0],traj[1] = (traj[0]-x_0)*numpy.cos(theta_0) + (traj[1]-y_0)\
        *numpy.sin(theta_0), -(traj[0]-x_0)*numpy.sin(theta_0) + \
        (traj[1]-y_0)*numpy.cos(theta_0)
    traj[2] = traj[2] - theta_0 
    return traj 


# Load reference trajectory
path = 'data/DdpResult/DdpResult_from_-3.928,1.353,-1.58_to_0,0,1.57_pos.dat'
traj = numpy.transpose(numpy.loadtxt(path))

# plt.subplot(1,2,1)
# plt.plot(traj[0],traj[1])

traj = translate(traj)
# print(traj[2])
# plt.subplot(1,2,2)
# plt.plot(traj[0],traj[1])
# plt.show()


velocity_ref = 0.25 # velocity we want the robot to walk

resized_traj = resizeTraj(traj, velocity_ref)

# instantiate pattern generator
nmpc    = NMPCGeneratorTraj(fsm_state='L/R')

# Pattern Generator Preparation
nmpc.   set_security_margin(0.09, 0.05)
# nmpc.   set_security_margin(0.04, 0.04) #on peut monter a v_ref=0.2

# instantiate plotter
show_canvas  = True
save_to_file = False
nmpc_p    = PlotterTraj(nmpc, traj, show_canvas, save_to_file)

raw_input("Press Enter to start")

# set initial values  
comx = [0.00679821, 0.0, 0.0]
comy = [0.08693283,0.0, 0.0]
comz = 8.92675352e-01
footx = 0.00949035
footy = 0.095
footq = 0.0

nmpc.   set_initial_values(comx, comy, comz, footx, footy, footq, foot='left')

interpolNmpc = Interpolation(0.001,nmpc)

sucess = True

f = open("data/nmpc_traj_offline.dat", "w")
f.write("")
f.close()

N = 16
T = 0.2
time_list = []

# Pattern Generator Event Loop
for i in range(N,len(resized_traj[0])):
    trajectory_reference = resized_traj[:,i-N:i]
    time_i = (i-N)*T

    start_time = time.time()

    nmpc.   set_trajectory_reference(trajectory_reference)

    # solve QP
    nb_failures = nmpc.   solve()
    time_list.append(time.time() - start_time)

    if nb_failures <= 2:
        nmpc.   simulate()
        interpolNmpc.interpolate(time_i)

        # initial value embedding by internal states and simulation
        comx, comy, comz, footx, footy, footq, foot, comq, future_footx,\
            future_footy, future_footq = \
        nmpc.update()
        nmpc.set_initial_values(comx, comy, comz, footx, footy, footq, foot, comq)

        if foot == "left":
            phase = 1
        else:
            phase = -1

        f = open("data/nmpc_traj_offline.dat", "a")
        line = str(comx[0])+ "  " + str(comx[1])+ "  " + str(comx[2])+ "  " +\
            str(comy[0])+ "  " + str(comy[1])+ "  " + str(comy[2])+ "  " +\
            str(comz)+ "  0  0  " + str(comq[0]) + "  " + str(comq[1]) + "  " +\
            str(comq[2]) + "  " + str(footx) + "  " + str(footy)+ "  " +\
            str(footq)+ "  " + str(phase)+ "  " + str(future_footx)+ "  " +\
            str(future_footy)+ "  " + str(future_footq) + " \n"
        f.write(line)
        f.close()

        if show_canvas:
            nmpc_p.update()
    else:
        sucess = False
        break

if sucess :
    print("Process terminated with sucess !!! :)")
else:
    print("Process terminated because of infeasible QP :'(")

nmpc.   data.save_to_file('./nmpc_traj_offline.json')

# print(time_list)
print("average time per iteration : ",numpy.mean(time_list))

show_canvas  = False
save_to_file = True

nmpc_p    = PlotterTraj(
    generator=None, trajectory=traj, show_canvas=show_canvas, save_to_file=save_to_file,
    filename='./nmpc_traj_offline',    fmt='pdf'
)

nmpc_p   .load_from_file('./nmpc_traj_offline.json')
nmpc_p   .update()
nmpc_p   .create_data_plot()

interpolNmpc.save_to_file("data/nmpc_traj_offline.csv")