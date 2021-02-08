#!/usr/bin/python

import numpy as np
import rospy
import crocoddyl
from math import pi, floor, sqrt, cos, sin, atan2
from scipy.optimize import minimize
from estimation.msg import TrajMsg
from std_msgs.msg import Bool
import time


#########################################################################
########################## FUNCTION DEFINITION	#########################
#########################################################################

def costFunction(weights,terminal_weights,x,u,posf):
	sum = 0
	for i in range (len(x)-1):
		sum += weights[0] + weights[1]*u[i][0]**2 + weights[2]*u[i][1]**2 + \
		weights[3]*u[i][2]**2 + weights[4]*(np.arctan2(posf[1]-x[i][1], \
		posf[0]-x[i][0]) - x[i][2])**2 
	sum += terminal_weights[0]*((posf[0]-x[-1][0])**2 + (posf[1]-x[-1][1])**2) +\
	terminal_weights[1]*(normalizeAngle(posf[2]-x[-1][2]))**2 +\
	terminal_weights[2]*(x[-1][3]**2 + x[-1][4]**2) +\
	terminal_weights[3]*x[-1][5]**2
	return sum

def optimizeT(T,x0,T_guess,model,terminal_model):
	T = int(T[0])
	if T > 0 and T < 2*T_guess:
		problem = crocoddyl.ShootingProblem(x0, [ model ] * T, terminal_model)
		ddp = crocoddyl.SolverDDP(problem)
		done = ddp.solve()
		# print(T,done,ddp.iter)
		if done:
			cost = costFunction(model.costWeights,terminal_model.costWeights,ddp.xs,ddp.us,model.finalState)/T
			# print(cost)
		elif ddp.iter < 50:
			cost = costFunction(model.costWeights,terminal_model.costWeights,ddp.xs,ddp.us,model.finalState)/T*10
		else:
			cost = 1e4
	else:
		cost = 1e8		
	return cost	

def translate(xs,x0):
	x,y,th = [],[],[]
	for state in xs:
		x.append(x0[0] + state[0])
		y.append(x0[1] + state[1])
		th.append(state[2])
	traj = [x,y,th]		
	return (traj)

def solveDdp(pos_i,pos_f):
	model = crocoddyl.ActionRunningModelHuman()
	data  = model.createData()
	model.costWeights = np.matrix([7.86951486e+00,4.00027971e+00,2.01459991e+01,\
		1.00000000e-06,9.99999967e+00]).T


	terminal_model = crocoddyl.ActionTerminalModelHuman()
	terminal_data  = terminal_model.createData()
	terminal_model.costWeights = np.matrix([9.98999939e+00,9.98999934e+00,\
		3.79999984e-01,3.35999389e+00]).T

	final_state = [(pos_f[0]-pos_i[0]),(pos_f[1]-pos_i[1]),pos_f[2]]
	model.finalState = np.matrix([final_state[0],final_state[1],final_state[2]]).T
	terminal_model.finalState = np.matrix([final_state[0],final_state[1],final_state[2]]).T
	init_state = np.matrix([ 0, 0,pos_i[2] , 0, 0, 0]).T

	distance = sqrt((pos_f[0]-pos_i[0])**2+(pos_f[1]-pos_i[1])**2)
	T_guess = int(distance*100/model.alpha*2/3)

	optimal = minimize(optimizeT, T_guess, args=(init_state,distance*100/model.alpha,\
		model,terminal_model),method='Nelder-Mead',options = {'xtol': 0.01,'ftol': 0.001})
	T_opt = int(optimal.x)

	problem = crocoddyl.ShootingProblem(init_state, T_opt*[model], terminal_model)
	ddp = crocoddyl.SolverDDP(problem)
	done = ddp.solve()
	traj = translate(ddp.xs, pos_i)
	# print("--- x ---",traj[0])
	# print("--- y ---",traj[1])

	return traj

def normalizeAngle(angle): 
	new_angle = angle
	while new_angle > pi:
		new_angle -= 2*pi
	while new_angle < -pi:
		new_angle += 2*pi
	return new_angle	

def solveEstimation(x,y,theta,N_OC,previous_est,new_initial_pos):
	# print("--- human traj x ---",x)
	# print("--- human traj y ---",y)
	# print("--- human traj th ---",theta)	
	init_model_list = []
	for i in range (0,len(x)):
		init_model = crocoddyl.ActionInitModelEstimation()
		init_data  = init_model.createData()
		init_model.costWeights = np.matrix([7.86951486e+00,4.00027971e+00,2.01459991e+01,\
		1.00000000e-06,10,10]).T
		init_model.currentState = np.matrix([x[i]-x[0],y[i]-y[0],theta[i]]).T
		init_model_list.append(init_model)

	model = crocoddyl.ActionRunningModelEstimation()
	data  = model.createData()
	model.costWeights = np.matrix([7.86951486e+00,4.00027971e+00,2.01459991e+01,\
		1.00000000e-06]).T
	# !!! reflechir fct ksi ici !!! 

	# print("len avt !!!",len(init_model_list))
	# print("len apres!!!",N_OC-len(x)+1)

	model_list = init_model_list + [model]*(N_OC-len(x)+1)

	if (len(previous_est[0]) == 0):
		pos_i = [x[0],y[0],theta[0]]		
		init_state = np.matrix([ 0, 0,pos_i[2] , 0, 0, 0]).T
		problem = crocoddyl.ShootingProblem(init_state, model_list[:-1] ,model_list[-1]) 
		ddp = crocoddyl.SolverDDP(problem)	
		done = ddp.solve()
	else:
		pos_i = new_initial_pos
		init_state = np.matrix([ 0, 0,pos_i[2] , 0, 0, 0]).T 
		# !!! verifier s il faut une vitesse non nulle ici !!! 
		problem = crocoddyl.ShootingProblem(init_state, model_list[:-1] ,model_list[-1]) 
		#  !!! verifier s il y a besoin d un terminal model ici !!! 

		# initial_guess = previous_est[1:len(previous_est)]
		# initial_guess.append(np.array([1,1,1,1,1,1]))
		ddp = crocoddyl.SolverDDP(problem)
		# done = ddp.solve(initial_guess)
		done = ddp.solve()
	print(done)
		# print("--- sol ddp ---",ddp.xs)

	traj = translate(ddp.xs, pos_i)
	# print("--- x ---",len(traj[0]))
	# print("--- y ---",traj[1])
	# print("--- th[:20] ---",traj[2][:20])	
	# print("--- th[20:] ---",traj[2][20:])
	return traj,ddp.xs,done

########################################################################
################################## MAIN ################################
########################################################################

# graph_disp = False

# model = crocoddyl.ActionRunningModelHuman()
# data  = model.createData()
# model.alpha = 1

# terminal_model = crocoddyl.ActionTerminalModelHuman()
# terminal_data  = terminal_model.createData()
# terminal_model.alpha = 1

# pos_f = [0,0,pi/2]
# pos_i = [3,1,0]
# model.costWeights = np.matrix([  7.86951486e+00,   4.00027971e+00,   2.01459991e+01,
# 							        1.00000000e-06,   9.99999967e+00,   9.98999939e+00,
# 							        9.98999934e+00,   3.79999984e-01,   3.35999389e+00]).T
# solveDdp(pos_i, pos_f,True)
# plt.show()

class estimation_pub:

	def __init__(self):
		self.sub = rospy.Subscriber("human_trajectory", TrajMsg, self.callback)
		self.pub = rospy.Publisher("estimated_trajectory", TrajMsg, queue_size=10)
		self.pub_ddp_solver_cv = rospy.Publisher("ddp_solver_cv", Bool, queue_size=10)

		# OC model parameters
		self.N_OC = rospy.get_param('N_OC')
		self.previous_sol = [[],[],[]]
		self.new_initial_pos = [0,0,0]

	def callback(self, traj):
		status = rospy.get_param('human_status')
		x, y, theta = traj.x_traj, traj.y_traj, traj.theta_traj
		estimated_traj = TrajMsg()

		if status == "Start":
			print("Start")
			est_traj = [[x[0]],[y[0]],[theta[0]]]

		if status == "Walk": 
			print("Walk",len(x))
			# print("--- old_pos_i ---",self.new_initial_pos)
			est_traj, sol, done = solveEstimation(x,y,theta,self.N_OC,\
				self.previous_sol,[x[0],y[0],theta[0]])#self.new_initial_pos)
			# self.new_initial_pos = [(est_traj[0][1]+est_traj[0][2])/2,\
			# (est_traj[1][1]+est_traj[1][2])/2,(est_traj[2][1]+est_traj[2][2])/2]
			self.new_initial_pos = [est_traj[0][2],est_traj[1][2],est_traj[2][2]]			
			self.previous_sol = sol
			self.pub_ddp_solver_cv.publish(done)
		if status == "Stop": #revoir ici, ajouter initial guess?
			print("Stop")
			pos_i = self.new_initial_pos
			pos_f = [x[0],y[0],theta[0]]
			est_traj = solveDdp(pos_i,pos_f)
			self.new_initial_pos = [est_traj[0][2],est_traj[1][2],est_traj[2][2]]

		estimated_traj.x_traj = est_traj[0]
		estimated_traj.y_traj = est_traj[1]
		estimated_traj.theta_traj = est_traj[2]
		# print(len(estimated_traj.x_traj),"--- x ---",estimated_traj.x_traj)
		# print(len(estimated_traj.y_traj),"--- y ---",estimated_traj.y_traj)	
		# print("--- new_pos_i ---",self.new_initial_pos)			
		self.pub.publish(estimated_traj)

if __name__ == '__main__':
	try:
		rospy.init_node('EstimationOC', anonymous=True)
		estimation_pub()
		while not rospy.is_shutdown():
			rospy.spin()
	except rospy.ROSInterruptException:
	    print("EstimationOC Shutting down")

