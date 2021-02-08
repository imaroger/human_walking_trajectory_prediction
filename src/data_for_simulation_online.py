from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import cos,sin
from mlp.bezier_predef import generateSmoothBezierTrajWithPredef
import pinocchio
from pinocchio import SE3, Quaternion
from pinocchio.rpy import rpyToMatrix
from mpl_toolkits.mplot3d import Axes3D

data = np.transpose(np.loadtxt("data/nmpc_traj_online.csv"))

# time_interp = np.linspace(0,100, len(traj[0]))

# plt.plot(time_interp, traj[0])
# plt.plot(time_interp, traj[3])
# plt.plot(time_com, com[0])
# plt.plot(time_com, com[1])

# plt.show()

# plt.plot(time_interp, traj[14])
# plt.plot(time_foot, footL[2])

# plt.show()

com_file = open("data/com.dat", "w")
am_file = open("data/am.dat", "w")
phase_file = open("data/phases.dat", "w")
footR_file = open("data/rightFoot.dat", "w")
footL_file = open("data/leftFoot.dat", "w")

first_DS = True

for i in range (len(data[0])):
	if first_DS and (data[14][i] != 0.105 or data[18][i] != 0.105):
		first_DS = False
	if not first_DS :
		com_file.write(str(data[0][i]) + "  " + str(data[3][i]) + "  0.892675352  " +\
			str(data[1][i]) + "  " + str(data[4][i])+ "  0.0  " +\
			str(data[2][i]) + "  " + str(data[5][i])+ "  0.0\n")

		am_file.write(str(data[6][i]) + "  " + str(data[7][i])+ "  " + str(data[8][i]) + "\n")

		rotR = rpyToMatrix(0,0,data[15][i])

		footR_file.write(str(data[12][i]) + "  " + str(data[13][i])+ "  " +\
			str(data[14][i]) + "  " + str(rotR[0][0]) + "  " +\
			str(rotR[0][1]) + "  " + str(rotR[0][2]) + "  " + str(rotR[1][0]) +\
			"  " + str(rotR[1][1]) + "  " + str(rotR[1][2])+ "  " +\
			str(rotR[2][0]) + "  " + str(rotR[2][1])+ "  " + str(rotR[2][2])+"\n")

		rotL = rpyToMatrix(0,0,data[19][i])

		footL_file.write(str(data[16][i]) + "  " + str(data[17][i])+ "  " +\
			str(data[18][i]) + "  " + str(rotL[0][0]) + "  " +\
			str(rotL[0][1]) + "  " + str(rotL[0][2]) + "  " + str(rotL[1][0]) +\
			"  " + str(rotL[1][1]) + "  " + str(rotL[1][2])+ "  " +\
			str(rotL[2][0]) + "  " + str(rotL[2][1])+ "  " + str(rotL[2][2])+"\n")

		if data[14][i] == 0.105 and data[18][i] == 0.105: # DS Phase
			# print(0)
			phase_file.write("0\n")
		elif data[14][i] != 0.105: # SS Phase : Left = Support foot
			# print(1)
			phase_file.write("1\n")	
		else :  # SS Phase : Right = Support foot
			# print(-1)			
			phase_file.write("-1\n")	

com_file.close()
am_file.close()
phase_file.close()
footR_file.close()
footL_file.close()

com = np.transpose(np.loadtxt("data/com.dat"))
footR = np.transpose(np.loadtxt("data/rightFoot.dat"))
footL = np.transpose(np.loadtxt("data/leftFoot.dat"))
phase = np.transpose(np.loadtxt("data/phases.dat"))
am = np.transpose(np.loadtxt("data/am.dat"))

plt.plot(com[0],com[1],color = 'blue',label='CoM')
plt.plot(footL[0],footL[1],color = 'green',label='Left Foot')
plt.plot(footR[0],footR[1],color = 'red',label='Right Foot')
legend = plt.legend()

plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(com[0],com[1],com[2],color = 'blue',label='CoM')
ax.plot3D(footL[0],footL[1],footL[2],color = 'green',label='Left Foot')
ax.plot3D(footR[0],footR[1],footR[2],color = 'red',label='Right Foot')
legend = plt.legend()

plt.show()

time_com = np.linspace(0,100,len(com[0]))
time_foot = np.linspace(0,100,len(footR[0]))

# plt.plot(time_foot,footR[2])
# plt.plot(time_foot,footL[2])
# plt.plot(time_foot,phase)
# plt.show()
# plt.plot(time_foot,footL[0])
# plt.plot(time_foot,footR[0])
# plt.plot(time_foot,footL[1])
# plt.plot(time_foot,footR[1])
# plt.show()
plt.plot(time_foot,np.arccos(footR[3]))
plt.plot(time_foot,np.arccos(footL[3]))
plt.plot(time_foot,phase)
plt.show()

time_reduced = np.arange(0,len(footR[0]),100)
arrow_len = 0.1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(footL[0],footL[1],footL[2],color = 'green',label='Left Foot')
ax.plot3D(footR[0],footR[1],footR[2],color = 'red',label='Right Foot')
legend = plt.legend()

ax.quiver(np.array(footR[0])[time_reduced],np.array(footR[1])[time_reduced],\
	np.array(footR[2])[time_reduced],np.array(footR[3])[time_reduced],\
	np.array(footR[6])[time_reduced],0, length=arrow_len, lw = 2)
ax.quiver(np.array(footL[0])[time_reduced],np.array(footL[1])[time_reduced],\
	np.array(footL[2])[time_reduced],np.array(footL[3])[time_reduced],\
	np.array(footL[6])[time_reduced],0, length=arrow_len, lw = 2)

ax.quiver(np.array(com[0])[time_reduced],np.array(com[1])[time_reduced],\
	np.array(com[2])[time_reduced],np.array(np.cos(am[0]))[time_reduced],\
	np.array(np.sin(am[0]))[time_reduced],0, length=arrow_len, lw = 2)


plt.show()