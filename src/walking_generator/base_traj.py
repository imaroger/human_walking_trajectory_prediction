import os, sys
import numpy
from math import cos, sin, sqrt
from copy import deepcopy

from helper import BaseTypeFoot, BaseTypeSupportFoot
from helper import ZMPState, CoMState
from visualization_traj import PlotData

class BaseGeneratorTraj(object):
    """
    Base class of walking pattern generator for humanoids, cf.
    LAAS-UHEI walking report.

    BaseClass provides all matrices and timestepping methods that are
    defined for the pattern generator family. In derived classes different
    problems and their solvers can be realized.
    """
    # define some constants
    g = 9.81

    # define list of members for plotting
    _plot_keys = [
        'time',
        'c_k_x',
        'c_k_y',
        'c_k_q',
        'f_k_x',
        'f_k_y',
        'f_k_q',
        'z_k_x',
        'z_k_y',
        'xi_k_x',
        'xi_k_y',
        'h_com',
        'C_kp1_x',
        'dC_kp1_x',
        'ddC_kp1_x',
        'C_kp1_y',
        'dC_kp1_y',
        'ddC_kp1_y',
        'C_kp1_q',
        'dC_kp1_q',
        'ddC_kp1_q',
        'dddC_k_x',
        'dddC_k_y',
        'dddC_k_q',
        'C_kp1_x_ref',
        'C_kp1_y_ref',
        'C_kp1_q_ref',
        'f_k_qL',
        'f_k_qR',
        'F_k_qL',
        'F_k_qR',
        'dF_k_qL',
        'dF_k_qR',
        'ddF_k_qL',
        'ddF_k_qR',
        'dddF_k_qL',
        'dddF_k_qR',
        'F_kp1_x',
        'F_kp1_y',
        'F_kp1_q',
        'F_k_x',
        'F_k_y',
        'F_k_q',
        'Z_kp1_x',
        'Z_kp1_y',
        'Xi_kp1_x',
        'Xi_kp1_y',        
        'fsm_state',
        'fsm_states',
    ]

    # define hull names for plotting
    _hull_keys = [
        'rfposhull',
        'lfposhull',
        'lfoot',
        'rfoot',
        'lfcophull',
        'rfcophull',
        'dscophull',
    ]

    # define values needed for calculations
    _data_keys = [
        'N',
        'nf',
        'T',
        'T_step',
        'footWidth',
        'footHeight',
        'footDistance',
    ]

    def __init__(
        self, N=16, T=0.1, T_step=0.8,
        fsm_state='D', fsm_sl=1
    ):
        """
        Initialize pattern generator, i.e.
        * allocate memory for matrices
        * populate them according to walking reference document

        Parameters
        ----------

        N : int
            Number of time steps of prediction horizon

        T : float
            Time increment between two time steps (default: 0.1 [s])

        nf : int
            Number of foot steps planned in advance (default: 2 [footsteps])

        T_step : float
            Time for the robot to make 1 step

        T_window : float
            Duration of the preview window of the controller
        fsm_state: str
            Initial state of the finite state machine for startin and stopping
            maneuvers.

        fsm_sl: int
            Number of steps in inplace stepping until stop
        """
        self.N = N
        self.T = T
        self.T_window = N*T
        self.T_step = T_step
        self.nf = (int)(self.T_window/T_step)
        self.time = 0.0
        # finite state machine for starting and landing maneuvers
        self._fsm_states = ('D', 'L/R', 'R/L', 'Lbar/Rbar', 'Rbar/Lbar')

        err_str = 'proposed state {} not in FSM states ({})'.format(fsm_state, self._fsm_states)
        assert fsm_state in self._fsm_states, err_str
        self.fsm_state = fsm_state
        self.fsm_states = numpy.array((self.fsm_state,)*self.nf, dtype=str)
        self._fsm_sl = fsm_sl

        # objective weights

        self.a = 1.0   # weight for CoM velocity tracking
        self.b = 0.0   # weight for CoM average velocity tracking
        self.c = 1e-06 # weight for ZMP reference tracking
        self.d = 1e-05 # weight for jerk minimization

        # center of mass initial values
        # NOTE they are not all equal to zero, because of half-sitting initial
        #      position of HRP-2 and the case, that its hip is not equal to its
        #      center of mass. z position is fixed because of LIPM approximation.


        self.c_k_x = numpy.array(
            (-1.98477637e-03, 0.0, 0.0), # 0.00124774 (HRP2) | -1.98477637e-03 (Pyrene)
            dtype=float
        )
        self.c_k_y = numpy.array(
            (7.22356707e-05, 0.0, 0.0), # 0.00157175(HRP2) | 7.22356707e-05 (Pyrene)
            dtype=float
        )
        self.c_k_q = numpy.zeros((3,), dtype=float)
        self.h_com = 8.92675352e-01 # 0.814 (HRP2) | 8.92675352e-01 (Pyrene)

        # center of mass matrices

        self.  C_kp1_x = numpy.zeros((N,), dtype=float)
        self. dC_kp1_x = numpy.zeros((N,), dtype=float)
        self.ddC_kp1_x = numpy.zeros((N,), dtype=float)

        self.  C_kp1_y = numpy.zeros((N,), dtype=float)
        self. dC_kp1_y = numpy.zeros((N,), dtype=float)
        self.ddC_kp1_y = numpy.zeros((N,), dtype=float)

        self.  C_kp1_q = numpy.zeros((N,), dtype=float)
        self. dC_kp1_q = numpy.zeros((N,), dtype=float)
        self.ddC_kp1_q = numpy.zeros((N,), dtype=float)

        # jerk controls for center of mass

        self.dddC_k_x = numpy.zeros((N,), dtype=float)
        self.dddC_k_y = numpy.zeros((N,), dtype=float)
        self.dddC_k_q = numpy.zeros((N,), dtype=float)

        # reference matrices

        # self. dC_kp1_x_ref = numpy.zeros((N,), dtype=float)
        # self. dC_kp1_y_ref = numpy.zeros((N,), dtype=float)
        # self. dC_kp1_q_ref = numpy.zeros((N,), dtype=float)
        # self.local_vel_ref = numpy.zeros((3,), dtype=float)

        self. C_kp1_x_ref = numpy.zeros((N,), dtype=float)
        self. C_kp1_y_ref = numpy.zeros((N,), dtype=float)
        self. C_kp1_q_ref = numpy.zeros((N,), dtype=float)
        self.traj_ref = numpy.zeros((3,N), dtype=float)


        # feet matrices
        # initial support foot positions and orientation
        # NOTE we assume the left foot to be the initial support foot
        self.f_k_x = 0.00949035
        self.f_k_y = 0.095
        self.f_k_q = 0.0

        self.F_k_x = numpy.zeros((self.nf,), dtype=float)
        self.F_k_y = numpy.zeros((self.nf,), dtype=float)
        self.F_k_q = numpy.zeros((self.nf,), dtype=float)

        # states for the foot orientation

        self.   f_k_qL = numpy.zeros((3,), dtype=float)
        self.   f_k_qR = numpy.zeros((3,), dtype=float)

        self.   F_k_qL = numpy.zeros((self.N,), dtype=float)
        self.   F_k_qR = numpy.zeros((self.N,), dtype=float)

        self.  dF_k_qL = numpy.zeros((self.N,), dtype=float)
        self.  dF_k_qR = numpy.zeros((self.N,), dtype=float)

        self. ddF_k_qL = numpy.zeros((self.N,), dtype=float)
        self. ddF_k_qR = numpy.zeros((self.N,), dtype=float)

        self.dddF_k_qL = numpy.zeros((self.N,), dtype=float)
        self.dddF_k_qR = numpy.zeros((self.N,), dtype=float)

        # foot angular velocity selection matrices objective
        # E_F = ( E_FR    0 )
        #       (    0 E_FL )
        # c est faux par rapport a la def en dessous...

        self.E_F  = numpy.zeros((self.N, 2*self.N), dtype=float)
        self.E_FR = self.E_F[:, :self.N]
        self.E_FL = self.E_F[:, self.N:]

        # foot angular velocity selection matrices objective

        self.E_F_bar  = numpy.zeros((self.N, 2*self.N), dtype=float)
        self.E_FR_bar = self.E_F_bar[:, :self.N]
        self.E_FL_bar = self.E_F_bar[:, self.N:]

        # zero moment point matrices

        self.z_k_x = self.c_k_x[0] - self.h_com/self.g * self.c_k_x[2]
        self.z_k_y = self.c_k_y[0] - self.h_com/self.g * self.c_k_y[2]

        self.Z_kp1_x = numpy.zeros((N,), dtype=float)
        self.Z_kp1_y = numpy.zeros((N,), dtype=float)

        # capture point matrices #NEW

        self.omega = sqrt(self.h_com/self.g)
        self.xi_k_x = self.c_k_x[0] + self.omega *self.c_k_x[1]
        self.xi_k_y = self.c_k_y[0] + self.omega * self.c_k_y[1]

        self.Xi_kp1_x = numpy.zeros((N,), dtype=float)
        self.Xi_kp1_y = numpy.zeros((N,), dtype=float)

        # transformation matrices

        self.Pps = numpy.zeros((N,3), dtype=float)
        self.Ppu = numpy.zeros((N,N), dtype=float)

        self.Pvs = numpy.zeros((N,3), dtype=float)
        self.Pvu = numpy.zeros((N,N), dtype=float)

        self.Pas = numpy.zeros((N,3), dtype=float)
        self.Pau = numpy.zeros((N,N), dtype=float)

        self.Pzs = numpy.zeros((N,3), dtype=float)
        self.Pzu = numpy.zeros((N,N), dtype=float)

        self.Pxis = numpy.zeros((N,3), dtype=float)
        self.Pxiu = numpy.zeros((N,N), dtype=float)

        # convex hulls used to bound the free placement of the foot
        self.nFootPosHullEdges = 5
            # support foot : right
        self.rfposhull = numpy.array((
                (-0.28, -0.2),
                (-0.20, -0.3),
                ( 0.00, -0.4),
                ( 0.20, -0.3),
                ( 0.28, -0.2),
        ), dtype=float)

            # support foot : left
        self.lfposhull = numpy.array((
                (-0.28, 0.2),
                (-0.20, 0.3),
                ( 0.00, 0.4),
                ( 0.20, 0.3),
                ( 0.28, 0.2),
        ), dtype=float)

        # set of Cartesian equalities
        self.A0r = numpy.zeros((5,2), dtype=float)
        self.ubB0r = numpy.zeros((5,), dtype=float)
        self.A0l = numpy.zeros((5,2), dtype=float)
        self.ubB0l = numpy.zeros((5,), dtype=float)

        # Linear constraints matrix
        self.nc_fchange_eq = 2
        self.eqAfoot = numpy.zeros((self.nc_fchange_eq,2*(self.N+self.nf)), dtype=float)
        self.eqBfoot = numpy.zeros((self.nc_fchange_eq,), dtype=float)

        # Linear constraints vector
        self.nc_foot_position = self.nf * self.nFootPosHullEdges
        self.Afoot   = numpy.zeros((self.nc_foot_position, 2*(self.N + self.nf)), dtype=float)
        self.lbBfoot =-numpy.ones((self.nc_foot_position), dtype=float)*1e+08
        self.ubBfoot = numpy.zeros((self.nc_foot_position), dtype=float)

        # security margins for CoP constraints
        self.SecurityMarginX = SMx = 0.09
        self.SecurityMarginY = SMy = 0.05

        # Position of the foot in the local foot frame
        self.nFootEdge    = 4
        self.footWidth    = 0.2 # 0.2172 (HRP2) | 0.2 (Pyrene)
        self.footHeight   = 0.12 # 0.1380 (HRP2) | 0.12 (Pyrene)
        self.footDistance = 0.19 # 0.2000 (HRP2) | 0.19 (Pyrene)

        # position of the vertices of the feet in the foot coordinates.
        # left foot
        self.lfoot = numpy.zeros((self.nFootEdge, 2), dtype=float)

        # right foot
        self.rfoot = numpy.zeros((self.nFootEdge, 2), dtype=float)

        # left foot
        self.lfcophull = numpy.zeros((self.nFootEdge, 2), dtype=float)

        # right foot
        self.rfcophull = numpy.zeros((self.nFootEdge, 2), dtype=float)

        # double support
        self.dscophull = numpy.zeros((self.nFootEdge, 2), dtype=float)

        # update hull arrays
        self._update_hulls()

        # Corresponding linear system from polygonal set
            # right foot
        self.A0rf   = numpy.zeros((self.nFootEdge,2), dtype=float)
        self.ubB0rf = numpy.zeros((self.nFootEdge,),  dtype=float)
            # left foot
        self.A0lf   = numpy.zeros((self.nFootEdge,2), dtype=float)
        self.ubB0lf = numpy.zeros((self.nFootEdge,),  dtype=float)
            # double support
        self.A0drf   = numpy.zeros((self.nFootEdge,2), dtype=float)
        self.ubB0drf = numpy.zeros((self.nFootEdge,),  dtype=float)
        self.A0dlf   = numpy.zeros((self.nFootEdge,2), dtype=float)
        self.ubB0dlf = numpy.zeros((self.nFootEdge,),  dtype=float)

        # transformation matrix for the CoP constraints in buildCoPconstraint()
        # constant in variables but varying in time, because of V_kp1
        # PzuV = ( PzuVx )
        #        ( PzuVy )
        #      = ( Pzu | -V_kp1 |   0 |      0 )
        #        (   0 |      0 | Pzu | -V_kp1 )

        self.PzuV  = numpy.zeros((2*self.N, 2*(self.N + self.nf)), dtype=float )
        self.PzuVx = self.PzuV[:self.N,:]
        self.PzuVy = self.PzuV[self.N:,:]

        # TODO tidy this up, because lots of redundant matrices
        # PzsC = ( PzsCx )
        #        ( PzsCy )
        #      = ( Pzs*c_k_x + v_kp1*f_k_x )
        #      = ( Pzs*c_k_y + v_kp1*f_k_y )
        self.PzsC  = numpy.zeros((2*self.N,), dtype=float )
        self.PzsCx = self.PzsC[:self.N]
        self.PzsCy = self.PzsC[self.N:]

        # v_kp1fc = ( v_kp1fc_x ) = ( v_kp1 * f_k_x)
        #           ( v_kp1fc_y )   ( v_kp1 * f_k_y)
        self.v_kp1fc = numpy.zeros((2*self.N,), dtype=float)
        self.v_kp1fc_x = self.v_kp1fc[:self.N]
        self.v_kp1fc_y = self.v_kp1fc[self.N:]

        # transformation matrix for the CP constraints in buildCoPconstraint()
        # constant in variables but varying in time, because of V_kp1
        # PxiuV = ( PxiuVx )
        #        ( PxiuVy )
        #      = ( Pxiu | -V_kp1 |   0 |      0 )
        #        (   0 |      0 | Pxiu | -V_kp1 )

        self.PxiuV  = numpy.zeros((2*self.N, 2*(self.N + self.nf)), dtype=float )
        self.PxiuVx = self.PxiuV[:self.N,:]
        self.PxiuVy = self.PxiuV[self.N:,:]

        # TODO tidy this up, because lots of redundant matrices
        # PxisC = ( PxisCx )
        #        ( PxisCy )
        #      = ( Pxis*c_k_x + v_kp1*f_k_x )
        #      = ( Pxis*c_k_y + v_kp1*f_k_y )
        self.PxisC  = numpy.zeros((2*self.N,), dtype=float )
        self.PxisCx = self.PxisC[:self.N]
        self.PxisCy = self.PxisC[self.N:]

        # D_kp1 = (D_kp1x, Dkp1_y)
        self.D_kp1  = numpy.zeros( (self.nFootEdge*self.N, 2*self.N), dtype=float )
        self.D_kp1x = self.D_kp1[:, :N] # view on big matrix
        self.D_kp1y = self.D_kp1[:,-N:] # view on big matrix
        self.b_kp1 = numpy.zeros( (self.nFootEdge*self.N,), dtype=float )

        # Constraint matrices
        self.nc_cop = self.N*self.nFootEdge
        self.Acop = numpy.zeros(
            (self.nc_cop, 2*(self.N+self.nf)),
             dtype=float
        )
        self.lbBcop = -numpy.ones((self.nc_cop), dtype=float)*1e+08
        self.ubBcop =  numpy.zeros((self.nc_cop), dtype=float)

        # Terminal Constraint : capture point
        self.nc_dcm = self.nFootEdge
        self.Adcm = numpy.zeros(
            (self.nc_dcm, 2*(self.N+self.nf)),
             dtype=float
        )
        self.lbBdcm = -numpy.ones((self.nc_dcm), dtype=float)*1e+08
        self.ubBdcm =  numpy.zeros((self.nc_dcm), dtype=float)

        # foot rotation constraints
        self.nc_fvel_eq = self.N # velocity constraints on support foot
        self.A_fvel_eq  = numpy.zeros((self.nc_fvel_eq, 2*self.N), dtype=float)
        self.B_fvel_eq  = numpy.zeros((self.nc_fvel_eq,), dtype=float)

        self.nc_fpos_ineq  = self.N # maximum orientation change
        self.A_fpos_ineq   = numpy.zeros((self.nc_fpos_ineq, 2*self.N), dtype=float)
        self.ubB_fpos_ineq = numpy.zeros((self.nc_fpos_ineq,), dtype=float)
        self.lbB_fpos_ineq = numpy.zeros((self.nc_fpos_ineq,), dtype=float)

        self.nc_fvel_ineq  = self.N # maximum angular velocity
        self.A_fvel_ineq   = numpy.zeros((self.nc_fvel_ineq, 2*self.N), dtype=float)
        self.ubB_fvel_ineq = numpy.zeros((self.nc_fvel_ineq,), dtype=float)
        self.lbB_fvel_ineq = numpy.zeros((self.nc_fvel_ineq,), dtype=float)

        # Current support state
        self.currentSupport = BaseTypeSupportFoot(x=self.f_k_x, y=self.f_k_y, theta=self.f_k_q, foot="left")
        self.currentSupport.timeLimit = 0
        self.currentSupport.ds = 0
        self.supportDeque = numpy.empty( (N,) , dtype=object )
        for i in range(N):
            self.supportDeque[i] = BaseTypeSupportFoot()
        self.supportDeque[0].ds = 1
        self.supportDeque[8].ds = 1

        """
        NOTE number of foot steps in prediction horizon changes between
        nf and nf+1, because when robot takes first step nf steps are
        planned on the prediction horizon, which makes a total of nf+1 steps.
        """
        self.v_kp1 = numpy.zeros((N,),   dtype=int)
        self.V_kp1 = numpy.zeros((N,self.nf), dtype=int)

        # initialize all elementary problem matrices, e.g.
        # state transformation matrices, constraints, etc.
        self._initialize_constant_matrices()
        self._initialize_cop_matrices()
        self._initialize_cp_matrices()
        self._initialize_selection_matrix()
        self._initialize_convex_hull_systems()

        self.data = PlotData(self)

    def _update_foot_selection_matrices(self):
        """ update the foot selection matrices E_F and E_F_bar """
        self.E_FR    [...] = 0.0
        self.E_FR_bar[...] = 0.0
        self.E_FL    [...] = 0.0
        self.E_FL_bar[...] = 0.0

        i = 0
        for j,supp in enumerate(self.supportDeque):
            if supp.foot == 'left':
                self.E_FR    [i,j] = 1.0
                self.E_FL    [i,j] = 0.0

                self.E_FR_bar[i,j] = 0.0
                self.E_FL_bar[i,j] = 1.0
            else:# supp.foot == 'right:'
                self.E_FR    [i,j] = 0.0
                self.E_FL    [i,j] = 1.0

                self.E_FR_bar[i,j] = 1.0
                self.E_FL_bar[i,j] = 0.0
            i += 1

    def _initialize_constant_matrices(self):
        """
        Initializes the constant transformation matrices, e.g. Pps, Ppu, Pvs,
        Pvu, Pas, Pau.
        """
        # renaming for convenience
        T_step = self.T_step
        T = self.T
        N = self.N
        nf = self.nf

        for i in range(N):
            j = i+1
            self.Pps[i, :] = (1.,   j*T,           (j**2*T**2)/2.)
            self.Pvs[i, :] = (0.,    1.,                      j*T)
            self.Pas[i, :] = (0.,    0.,                       1.)

            for j in range(N):
                if j <= i:
                    self.Ppu[i, j] = (3.*(i-j)**2 + 3.*(i-j) + 1.)*T**3/6.
                    self.Pvu[i, j] = (2.*(i-j) + 1.)*T**2/2.
                    self.Pau[i, j] = T

    def _initialize_cop_matrices(self):
        # """
        # Initialize center of pressure matrices, which are dependent on current
        # height of center of mass (self.h_com).
        # """
        # # renaming for convenience
        # T_step = self.T_step
        # T = self.T
        # N = self.N
        # nf = self.nf
        # h_com = self.h_com
        # g = self.g

        # for i in range(N):
        #     j = i+1
        #     self.Pzs[i, :] = (1.,   j*T, (j**2*T**2)/2. - h_com/g)

        #     for j in range(N):
        #         if j <= i:
        #             self.Pzu[i, j] = (3.*(i-j)**2 + 3.*(i-j) + 1.)*T**3/6. - T*h_com/g
        
        #NEW
        self.Pzs = self.Pps - self.omega**2 * self.Pas
        self.Pzu = self.Ppu - self.omega**2 * self.Pau

    def _initialize_cp_matrices(self):
        """
        Initialize capture point matrices, which are dependent on current
        height of center of mass (self.h_com).
        """
        self.Pxis = self.Pps + 1./self.omega * self.Pvs
        self.Pxiu = self.Ppu + 1./self.omega * self.Pvu       

    def _initialize_selection_matrix(self):
        """ Initialize selection vector and matrix. """
        # renaming for convenience
        T_step = self.T_step
        T = self.T
        N = self.N
        nf = self.nf

        # initialize foot decision vector and matrix
        nstep = int(self.T_step/T) # time span of single support phase
        self.v_kp1[:nstep] = 1 # definitions of initial support leg

        for j in range (nf):
            a = min((j+1)*nstep, N)
            b = min((j+2)*nstep, N)
            self.V_kp1[a:b,j] = 1

        self._calculate_support_order()

    def _update_hulls(self):
        """ update shape polygon of convex hulls """
        # renaming for convenience
        fW = self.footWidth
        fH = self.footHeight
        fD = self.footDistance

        SMx = self.SecurityMarginX
        SMy = self.SecurityMarginY

        #  #<---footWidth---># --- ---
        #  #<>=SMx     SMx=<>#  |   |=SMy
        #  #  *-----------*  #  |  ---
        #  #  |           |  #  |=footHeight
        #  #  *-----------*  #  |  ---
        #  #                 #  |   |=SMy
        #  #-----------------# --- ---

        # left foot
        self.lfoot[0,:] =  0.5*fW,  0.5*fH
        self.lfoot[1,:] =  0.5*fW, -0.5*fH
        self.lfoot[2,:] = -0.5*fW, -0.5*fH
        self.lfoot[3,:] = -0.5*fW,  0.5*fH

        # right foot
        self.rfoot[0,:] =  0.5*fW, -0.5*fH
        self.rfoot[1,:] =  0.5*fW,  0.5*fH
        self.rfoot[2,:] = -0.5*fW,  0.5*fH
        self.rfoot[3,:] = -0.5*fW, -0.5*fH

        # left foot
        self.lfcophull[0,:] =  (0.5*fW - SMx),  (0.5*fH - SMy)
        self.lfcophull[1,:] =  (0.5*fW - SMx), -(0.5*fH - SMy)
        self.lfcophull[2,:] = -(0.5*fW - SMx), -(0.5*fH - SMy)
        self.lfcophull[3,:] = -(0.5*fW - SMx),  (0.5*fH - SMy)

        # right foot
        self.rfcophull[0,:] =  (0.5*fW - SMx), -(0.5*fH - SMy)
        self.rfcophull[1,:] =  (0.5*fW - SMx),  (0.5*fH - SMy)
        self.rfcophull[2,:] = -(0.5*fW - SMx),  (0.5*fH - SMy)
        self.rfcophull[3,:] = -(0.5*fW - SMx), -(0.5*fH - SMy)

        # double support
        # |<----d---->| d = 2*footHeight + footDistance
        # |-----------|
        # | *   *   * |
        # |-^-------^-|
        # left     right
        # foot     foot
        self.dscophull[0,:] =  (0.5*fW - SMx),  (0.5*(fD + fH) - SMy)
        self.dscophull[1,:] =  (0.5*fW - SMx), -(0.5*(fD + fH) - SMy)
        self.dscophull[2,:] = -(0.5*fW - SMx), -(0.5*(fD + fH) - SMy)
        self.dscophull[3,:] = -(0.5*fW - SMx),  (0.5*(fD + fH) - SMy)

    def _update_selection_matrices(self):
        """
        Update selection vector v_kp1 and selection matrix V_kp1.

        Therefore shift foot decision vector and matrix by one row up,
        i.e. the first entry in the selection vector and the first row in the
        selection matrix drops out and selection vector's dropped first value
        becomes the last entry in the decision matrix
        """
        nf = self.nf
        nstep = int(self.T_step/self.T)
        N = self.N

        # save first value for concatenation
        first_entry_v_kp1 = self.v_kp1[0].copy()

        self.v_kp1[:-1]   = self.v_kp1[1:]
        self.V_kp1[:-1,:] = self.V_kp1[1:,:]

        # clear last row
        self.V_kp1[-1,:] = 0

        # concatenate last entry
        self.V_kp1[-1, -1] = first_entry_v_kp1

        # when first column of selection matrix becomes zero,
        # then shift columns by one to the front
        if (self.v_kp1 == 0).all():
            self.v_kp1[:] = self.V_kp1[:,0]
            self.V_kp1[:,:-1] = self.V_kp1[:,1:]
            self.V_kp1[:,-1] = 0

            # update support foot
            self.f_k_x = self.F_k_x[0]
            self.f_k_y = self.F_k_y[0]
            self.f_k_q = self.F_k_q[0]

            # print("---",self.f_k_x,"---")

            self.currentSupport.x = self.f_k_x
            self.currentSupport.y = self.f_k_y
            self.currentSupport.q = self.f_k_q

            if self.currentSupport.foot == 'right':
                self.currentSupport.foot  = 'left'
            else:
                self.currentSupport.foot = 'right'

            # update support order with new foot state
            self._calculate_support_order()

            """
            # also update finite state machine
            self.fsm_state = self.fsm_states[0].copy()
            self.fsm_states[:-1] = self.fsm_states[1:]

            # TODO How to update last entry in FSM?
            # if any reference velocity is given and the CoM is moving then
            # the next step last optimized step should be moving
            if (self.dC_kp1_x_ref != 0.0).any() \
            or (self.dC_kp1_y_ref != 0.0).any() \
            or (self.dC_kp1_q_ref != 0.0).any():
                if (self.c_k_x[1:] != 0.0).any() \
                or (self.c_k_y[1:] != 0.0).any() \
                or (self.c_k_q[1:] != 0.0).any():
                    if self.supportDeque[-1].foot == 'right':
                        self.fsm_states[-1] = 'L/R'
                    else:
                        self.fsm_states[-1] = 'R/L'
            # else stay in double support
            else:
                self.fsm_states[-1] = 'D'
            """

    def _initialize_convex_hull_systems(self):
        # linear system corresponding to the convex hulls
        self.ComputeLinearSystem( self.rfposhull, "right", self.A0r, self.ubB0r)
        self.ComputeLinearSystem( self.lfposhull, "left", self.A0l, self.ubB0l)

        # linear system corresponding to the convex hulls
            # right foot
        self.ComputeLinearSystem( self.rfcophull,  "right", self.A0rf, self.ubB0rf)
            # left foot
        self.ComputeLinearSystem( self.lfcophull,  "left",  self.A0lf, self.ubB0lf)
            # double support
            # NOTE hull has to be shifted by half of feet distance in y direction
        self.ComputeLinearSystem(self.dscophull, "left",  self.A0dlf, self.ubB0dlf)
        self.ComputeLinearSystem(self.dscophull, "right", self.A0drf, self.ubB0drf)

    def ComputeLinearSystem(self, hull, foot, A0, B0 ):
        """
        automatically calculate linear constraints from polygon description
        """
        # get number of edged from the hull specification, e.g.
        # single and double support polygon, foot position hull
        nEdges = hull.shape[0]

        # get sign for hull from given foot
        if foot == "left" :
            sign = 1
        else :
            sign = -1

        # calculate linear constraints from hull
        # walk around polygon and calculate coordinate representation of
        # constraints
        for i in range(nEdges):
            # special case for first and last entry in hull
            if i == nEdges-1 :
                k = 0
            else :
                k = i + 1

            # rename point coordinates for convenience
            x1 = hull[i,0]
            y1 = hull[i,1]
            x2 = hull[k,0]
            y2 = hull[k,1]

            # calculate support vectors
            dx = y1 - y2
            dy = x2 - x1
            dc = dx*x1 + dy*y1

            # symmetrical constraints
            A0[i,0] = sign * dx
            A0[i,1] = sign * dy
            B0[i] =   sign * dc

    def _calculate_support_order(self):
        self.currentSupport.ds = deepcopy(self.supportDeque[0].ds)

        # find correct initial support foot
        if (self.currentSupport.foot == "left" ) :
            pair = "left"
            impair = "right"
        else :
            pair = "right"
            impair = "left"

        # define support feet for whole horizon
        for i in range(self.N):
            if self.v_kp1[i] == 1:
                self.supportDeque[i].foot = self.currentSupport.foot
                self.supportDeque[i].stepNumber = 0
            else:
                for j in range(self.nf):
                    if self.V_kp1[i][j] == 1 :
                        self.supportDeque[i].stepNumber = j+1
                        if (j % 2) == 1:
                            self.supportDeque[i].foot = pair
                        else :
                            self.supportDeque[i].foot = impair

        if numpy.sum(self.v_kp1)==8 :
            self.supportDeque[0].ds = 1
        else :
            self.supportDeque[0].ds = 0
        for i in range (1,self.N):
            self.supportDeque[i].ds = self.supportDeque[i].stepNumber - self.supportDeque[i-1].stepNumber

        timeLimit = self.supportDeque[0].timeLimit
        for i in range(self.N):
            if self.supportDeque[i].ds == 1 :
                timeLimit = timeLimit + self.T_step
            self.supportDeque[i].timeLimit = timeLimit

    def set_security_margin(self, margin_x = 0.04, margin_y=0.04):
        """
        define security margins for constraints CoP constraints

        .. NOTE: Will recreate constraint matrices

        Parameters
        ----------

        margin_x: 0 < float < footWidth
            security margin to narrow center of pressure constraints in x direction

        margin_y: 0 < float < footWidth
            security margin to narrow center of pressure constraints in x direction
        """
        self.SecurityMarginX = margin_x
        self.SecurityMarginY = margin_y

        # update cop hull
        self._update_hulls()

        # rebuild cop constraints
        self._initialize_convex_hull_systems()

        # rebuild constraints
        self.buildConstraints()

    # def set_velocity_reference(self,local_vel_ref):
    #     """
    #     Velocity reference update and computed from a local frame to a global frame using the
    #     current support foot frame

    #     Parameters
    #     ----------

    #     vel_ref: [dx,dy,dq]
    #         reference velocity in x, y and q
    #     """
    #     # get feet orientation states from feet jerks
    #     self.local_vel_ref = local_vel_ref
    #     self.  F_kp1_qL = self.Pps.dot(self.f_k_qL) + self.Ppu.dot(self.dddF_k_qL)
    #     self.  F_kp1_qR = self.Pps.dot(self.f_k_qR) + self.Ppu.dot(self.dddF_k_qR)

    #     flyingFoot = self.E_FR.dot(self.F_kp1_qR) + self.E_FL.dot(self.F_kp1_qL)
    #     supportFoot = self.E_FR_bar.dot(self.F_kp1_qR) + self.E_FL_bar.dot(self.F_kp1_qL)
    #     q = (flyingFoot[0] + supportFoot[0])*0.5
    #     self.dC_kp1_x_ref[...] = deepcopy( local_vel_ref[0] * cos(q) - local_vel_ref[1] * sin(q) )
    #     self.dC_kp1_y_ref[...] = deepcopy( local_vel_ref[0] * sin(q) + local_vel_ref[1] * cos(q) )
    #     self.dC_kp1_q_ref[...] = deepcopy( local_vel_ref[2] )

    #     print self.dC_kp1_x_ref[...]
    #     print self.dC_kp1_y_ref[...]
    #     print self.dC_kp1_q_ref[...]

    def set_trajectory_reference(self,traj_ref):

        self.traj_ref = traj_ref
        self.C_kp1_x_ref[...] = deepcopy(traj_ref[0])
        self.C_kp1_y_ref[...] = deepcopy(traj_ref[1])
        self.C_kp1_q_ref[...] = deepcopy(traj_ref[2])

        # print self.C_kp1_x_ref[...]
        # print self.C_kp1_y_ref[...]
        # print self.C_kp1_q_ref[...]        

    def set_initial_values(self,
        com_x, com_y , com_z,
        foot_x, foot_y, foot_q, foot='left',
        com_q=(0,0,0)
    ):
        """
        initial value embedding for pattern generator, i.e. each iteration of
        pattern generator differs in:

        * initial com state, i.e. com_a = [c_a, dc_a, ddc_a], a in {x,y,q}
        * initials support foot setup, i.e.

        .. NOTE: Will recreate constraint matrices, support order and
                 transformation matrices.

        Parameters
        ----------

        com_x: [pos, vec, acc]
            current x position, velocity and acceleration of center of mass

        com_y: [pos, vec, acc]
            current y position, velocity and acceleration of center of mass

        com_z: float
            current z position of center of mass

        foot_x: float
            current x position of support foot

        foot_y: float
            current y position of support foot

        foot_q: float
            current orientation of support foot

        foot: str
            tells actual support foot state, i.e. 'right' or by default 'left'

        com_q: [ang, vec, acc]
            current orientation, angular velocity and acceleration of center of mass

        """
        # update CoM states
        self.c_k_x[...] = com_x
        self.c_k_y[...] = com_y

        if not self.h_com == com_z:
            self.h_com = com_z
            self._initialize_cop_matrices()

        # update support foot if necessary
        newSupport = BaseTypeSupportFoot(x=foot_x, y=foot_y, theta=foot_q, foot=foot)
        if self.currentSupport != newSupport \
        or self.f_k_x != foot_x \
        or self.f_k_y != foot_y \
        or self.f_k_q != foot_q :
            # take newSupport as current support
            self.currentSupport = newSupport

            # update support foot states
            self.f_k_x = foot_x
            self.f_k_y = foot_y
            self.f_k_q = foot_q

        # always recalculate support order
        self._calculate_support_order()

        # update current CoP values
        self.z_k_x = self.c_k_x[0] - self.h_com/self.g * self.c_k_x[2]
        self.z_k_y = self.c_k_y[0] - self.h_com/self.g * self.c_k_y[2]

        # update current CP values
        self.xi_k_x = self.c_k_x[0] + self.omega * self.c_k_x[1]
        self.xi_k_y = self.c_k_y[0] + self.omega * self.c_k_y[1]       

        # rebuild all constraints
        self.buildConstraints()

    def update(self):
        """
        Update all interior matrices, vectors.
        Has to be used to prepare the QP after each iteration
        """

        # after solution simulate to get current states on horizon
        self.simulate()

        # update internal data
        self._update_data()

        # update internal time
        self.time += self.T

        # update matrices
        oldSupport = deepcopy(self.currentSupport)
        self._update_selection_matrices()
        if self.currentSupport != oldSupport \
        or self.f_k_x != self.currentSupport.x \
        or self.f_k_y != self.currentSupport.y \
        or self.f_k_q != self.currentSupport.q :
            raise NotImplementedError

        # provide copy of updated states as return value
        f_k_x = deepcopy(self.f_k_x)
        f_k_y = deepcopy(self.f_k_y)
        f_k_q = deepcopy(self.f_k_q)
        foot  = deepcopy(self.currentSupport.foot)

        # get data for initialization of next iteration
        c_k_x = numpy.zeros((3,), dtype=float)
        c_k_x[0] = self.  C_kp1_x[0]
        c_k_x[1] = self. dC_kp1_x[0]
        c_k_x[2] = self.ddC_kp1_x[0]

        c_k_y = numpy.zeros((3,), dtype=float)
        c_k_y[0] = self.  C_kp1_y[0]
        c_k_y[1] = self. dC_kp1_y[0]
        c_k_y[2] = self.ddC_kp1_y[0]

        c_k_q = numpy.zeros((3,), dtype=float)
        c_k_q[0] = self.  C_kp1_q[0]
        c_k_q[1] = self. dC_kp1_q[0]
        c_k_q[2] = self.ddC_kp1_q[0]

        # left foot
        f_k_qR = numpy.zeros((3,), dtype=float)
        f_k_qR[0] = self.  F_kp1_qR[0]
        f_k_qR[1] = self. dF_kp1_qR[0]
        f_k_qR[2] = self.ddF_kp1_qR[0]
        self.f_k_qR[...] = f_k_qR

        f_k_qL = numpy.zeros((3,), dtype=float)
        f_k_qL[0] = self.  F_kp1_qL[0]
        f_k_qL[1] = self. dF_kp1_qL[0]
        f_k_qL[2] = self.ddF_kp1_qL[0]
        self.f_k_qL[...] = f_k_qL

        if self.currentSupport.foot == "left" :
            self.f_k_q = self.f_k_qL[0]
        else :
            self.f_k_q = self.f_k_qR[0]
        self.currentSupport.q = self.f_k_q

        self.set_trajectory_reference(self.traj_ref)

        # # ZMP_k and CP_k
        # zmp_k_x = self.  Z_kp1_x[0]
        # zmp_k_y = self.  Z_kp1_y[0]
        # cp_k_x = self.  Xi_kp1_x[0]
        # cp_k_y = self.  Xi_kp1_y[0]

        # # ZMP_k+N and CP_k+N
        # zmp_N_x = self.  Z_kp1_x[-1]
        # zmp_N_y = self.  Z_kp1_y[-1]       
        # cp_N_x = self.  Xi_kp1_x[-1]
        # cp_N_y = self.  Xi_kp1_y[-1]

        # # CoM_k+N
        # c_N_x = numpy.zeros((3,), dtype=float)
        # c_N_x[0] = self.  C_kp1_x[-1]
        # c_N_x[1] = self. dC_kp1_x[-1]
        # c_N_x[2] = self.ddC_kp1_x[-1]

        # c_N_y = numpy.zeros((3,), dtype=float)
        # c_N_y[0] = self.  C_kp1_y[-1]
        # c_N_y[1] = self. dC_kp1_y[-1]
        # c_N_y[2] = self.ddC_kp1_y[-1]

        # Future steps

        if(numpy.sum(self.v_kp1) != 8):
            F_k_x, F_k_y = self.F_k_x[0], self.F_k_y[0]
        else :
            F_k_x, F_k_y = self.F_k_x[1], self.F_k_y[1]
        # print("------")
        # print(self.v_kp1,numpy.sum(self.v_kp1))
        # print("actual",f_k_x,f_k_q,self.f_k_q,foot)
        # print(f_k_qR[0],f_k_qL[0])
        # print("future",self.F_k_x)
        # print(self.F_kp1_qR,self.F_kp1_qL)

        if self.currentSupport.foot == "left" :
            F_k_q = self.F_kp1_qR[numpy.sum(self.v_kp1)-1]
        else:
            F_k_q = self.F_kp1_qL[numpy.sum(self.v_kp1)-1]
        # print(F_k_q)

        return c_k_x, c_k_y, self.h_com, f_k_x, f_k_y, f_k_q, foot,\
            c_k_q, F_k_x, F_k_y, F_k_q

    def _update_data(self):
        self.data.update()

    def simulate(self):
        """
        integrates model for given initial CoM states, jerks and feet positions
        and orientations by applying the linear time stepping scheme
        """
        # get CoM states from jerks
        self.  C_kp1_x = self.Pps.dot(self.c_k_x) + self.Ppu.dot(self.dddC_k_x)
        self. dC_kp1_x = self.Pvs.dot(self.c_k_x) + self.Pvu.dot(self.dddC_k_x)
        self.ddC_kp1_x = self.Pas.dot(self.c_k_x) + self.Pau.dot(self.dddC_k_x)

        self.  C_kp1_y = self.Pps.dot(self.c_k_y) + self.Ppu.dot(self.dddC_k_y)
        self. dC_kp1_y = self.Pvs.dot(self.c_k_y) + self.Pvu.dot(self.dddC_k_y)
        self.ddC_kp1_y = self.Pas.dot(self.c_k_y) + self.Pau.dot(self.dddC_k_y)

        # get feet orientation states from feet jerks
        self.  F_kp1_qL = self.Pps.dot(self.f_k_qL) + self.Ppu.dot(self.dddF_k_qL)
        self. dF_kp1_qL = self.Pvs.dot(self.f_k_qL) + self.Pvu.dot(self.dddF_k_qL)
        self.ddF_kp1_qL = self.Pas.dot(self.f_k_qL) + self.Pau.dot(self.dddF_k_qL)

        self.  F_kp1_qR = self.Pps.dot(self.f_k_qR) + self.Ppu.dot(self.dddF_k_qR)
        self. dF_kp1_qR = self.Pvs.dot(self.f_k_qR) + self.Pvu.dot(self.dddF_k_qR)
        self.ddF_kp1_qR = self.Pas.dot(self.f_k_qR) + self.Pau.dot(self.dddF_k_qR)

        self.  C_kp1_q = 0.5 * ( self.  F_kp1_qL + self.  F_kp1_qR )
        self. dC_kp1_q = 0.5 * ( self. dF_kp1_qL + self. dF_kp1_qR )
        self.ddC_kp1_q = 0.5 * ( self.ddF_kp1_qL + self.ddF_kp1_qR )

        # get support foot orientation
        self.F_kp1_q = self.E_FR_bar.dot(self.F_kp1_qR) \
                     + self.E_FL_bar.dot(self.F_kp1_qL)

        for j in range(self.nf):
            for i in range(self.N):
                if self.V_kp1[i,j] != 0:
                    self.F_k_q[j] = self.F_kp1_q[i]
                    break
            else:
                self.F_k_q[j] = 0.0

        # get ZMP states from jerks
        self.Z_kp1_x = self.Pzs.dot(self.c_k_x) + self.Pzu.dot(self.dddC_k_x)
        self.Z_kp1_y = self.Pzs.dot(self.c_k_y) + self.Pzu.dot(self.dddC_k_y)

        # get CP states from jerks
        self.Xi_kp1_x = self.Pxis.dot(self.c_k_x) + self.Pxiu.dot(self.dddC_k_x)
        self.Xi_kp1_y = self.Pxis.dot(self.c_k_y) + self.Pxiu.dot(self.dddC_k_y)       

    def buildConstraints(self):
        """
        builds constraint matrices for solver

        NOTE problems are assembled in the solver implementations
        """
        self.buildCoPconstraint()
        self.buildFootEqConstraint()
        self.buildFootIneqConstraint()
        self.buildFootRotationConstraints()
        self.buildRotIneqConstraint()

    def _update_cop_constraint_transformation(self):
        """ update foot constraint transformation matrices. """
        # every time instant in the pattern generator constraints
        # depend on the support order
        theta_vec = [self.f_k_q,self.F_k_q[0],self.F_k_q[1]]
        for i in range(self.N):
            theta = theta_vec[self.supportDeque[i].stepNumber]
            rotMat = numpy.array([[cos(theta), sin(theta)],[-sin(theta), cos(theta)]])
            if self.supportDeque[i].foot == "left" :
                A0 = self.A0lf.dot(rotMat)
                B0 = self.ubB0lf
                D0 = self.A0dlf.dot(rotMat)
                d0 = self.ubB0dlf
            else :
                A0 = self.A0rf.dot(rotMat)
                B0 = self.ubB0rf
                D0 = self.A0drf.dot(rotMat)
                d0 = self.ubB0drf

            # get support foot and check if it is double support
            for j in range(self.nf):
                if self.V_kp1[i,j] == 1:
                    if self.fsm_states[j] == 'D':
                        A0 = D0
                        B0 = d0
                else:
                    pass

            for k in range(self.nFootEdge):
                # get d_i+1^x(f^theta)
                self.D_kp1x[i*self.nFootEdge+k, i] = A0[k][0]
                # get d_i+1^y(f^theta)
                self.D_kp1y[i*self.nFootEdge+k, i] = A0[k][1]
                # get right hand side of equation
                self.b_kp1 [i*self.nFootEdge+k]    = B0[k]

    def buildCoPconstraint(self):
        """
        build the constraint enforcing the center of pressure to stay inside
        the support polygon given through the convex hull of the foot.
        """
        # change entries according to support order changes in D_kp1
        self._update_cop_constraint_transformation()

        #rename for convenience
        D_kp1 = self.D_kp1
        PzuV  = self.PzuV
        PzuVx = self.PzuVx
        PzuVy = self.PzuVy
        PzsC  = self.PzsC
        PzsCx = self.PzsCx
        PzsCy = self.PzsCy  
        v_kp1fc   = self.v_kp1fc
        v_kp1fc_x = self.v_kp1fc_x
        v_kp1fc_y = self.v_kp1fc_y

        # build constraint transformation matrices
        # PzuV = ( PzuVx )
        #        ( PzuVy )

        # PzuVx = ( Pzu | -V_kp1 |   0 |      0 )
        PzuVx[:,      :self.N        ] =  self.Pzu # TODO this is constant in matrix and should go into the build up matrice part
        PzuVx[:,self.N:self.N+self.nf] = -self.V_kp1

        # PzuVy = (   0 |      0 | Pzu | -V_kp1 )
        PzuVy[:,-self.N-self.nf:-self.nf] =  self.Pzu # TODO this is constant in matrix and should go into the build up matrice part
        PzuVy[:,       -self.nf:       ] = -self.V_kp1

        # PzuV = ( PzsCx ) = ( Pzs * c_k_x)
        #        ( PzsCy )   ( Pzs * c_k_y)
        PzsCx[...] = self.Pzs.dot(self.c_k_x) #+ self.v_kp1.dot(self.f_k_x)
        PzsCy[...] = self.Pzs.dot(self.c_k_y) #+ self.v_kp1.dot(self.f_k_y)
   

        # v_kp1fc = ( v_kp1fc_x ) = ( v_kp1 * f_k_x)
        #           ( v_kp1fc_y )   ( v_kp1 * f_k_y)
        v_kp1fc_x[...] = self.v_kp1.dot(self.f_k_x)
        v_kp1fc_y[...] = self.v_kp1.dot(self.f_k_y)

        # build CoP linear constraints
        # NOTED_kp1x D_kp1 is member and D_kp1 = ( D_kp1x | D_kp1y )
        #      D_kp1x,y contains entries from support polygon

        self.Acop[...]   = D_kp1.dot(PzuV)
        self.ubBcop[...] = self.b_kp1 - D_kp1.dot(PzsC) + D_kp1.dot(v_kp1fc)

        PxiuV  = self.PxiuV
        PxiuVx = self.PxiuVx
        PxiuVy = self.PxiuVy
        PxisC  = self.PxisC
        PxisCx = self.PxisCx
        PxisCy = self.PxisCy

        # build constraint transformation matrices
        # PxiuV = ( PxiuVx )
        #        ( PxiuVy )

        # PxiuVx = ( Pxiu | -V_kp1 |   0 |      0 )
        PxiuVx[:,      :self.N        ] =  self.Pxiu # TODO this is constant in matrix and should go into the build up matrice part
        PxiuVx[:,self.N:self.N+self.nf] = -self.V_kp1

        # PzuVy = (   0 |      0 | Pzu | -V_kp1 )
        PxiuVy[:,-self.N-self.nf:-self.nf] =  self.Pxiu # TODO this is constant in matrix and should go into the build up matrice part
        PxiuVy[:,       -self.nf:       ] = -self.V_kp1

        # PxiuV = ( PxisCx ) = ( Pxis * c_k_x)
        #        ( PxisCy )   ( Pxis * c_k_y)
        PxisCx[...] = self.Pxis.dot(self.c_k_x) #+ self.v_kp1.dot(self.f_k_x)
        PxisCy[...] = self.Pxis.dot(self.c_k_y) #+ self.v_kp1.dot(self.f_k_y)

        # # build CP linear constraints

        self.Adcm[...]   = D_kp1[-self.nFootEdge:,:].dot(PxiuV)
        self.ubBdcm[...] = self.b_kp1[-self.nFootEdge:] - D_kp1[-self.nFootEdge:,:].dot(PxisC) + D_kp1[-self.nFootEdge:,:].dot(v_kp1fc)

        # print("cop",self.Acop[-self.nFootEdge:,:])
        # print("dcm",self.Adcm)

    def buildFootEqConstraint(self):
        """[-self.nFootEdge
        create constraints that freezes foot position optimization when swing
        foot comes close to foot step in preview window. Needed for proper
        interpolation of trajectory.
        """
        # B <= A x <= B
        # Support_Foot(k+1) = Support_Foot(k)
        itBeforeLanding = numpy.sum(self.v_kp1)
        itBeforeLandingThreshold = 3
        if ( itBeforeLanding < itBeforeLandingThreshold ) :
            self.eqAfoot[0,   self.N        ] = 1.
            self.eqAfoot[1, 2*self.N+self.nf] = 1.

            self.eqBfoot[0] = self.F_k_x[0]
            self.eqBfoot[1] = self.F_k_y[0]
        else:
            self.eqAfoot[0,   self.N        ] = 0.0
            self.eqAfoot[1, 2*self.N+self.nf] = 0.0

            self.eqBfoot[0] = 0.0
            self.eqBfoot[1] = 0.0

    def buildFootIneqConstraint(self):
        """
        build linear inequality constraints for the placement of the feet

        NOTE: needs actual self.supportFoot to work properly
        """
        # inequality constraint on both feet A u + B <= 0
        # A0 R(theta) [Fx_k+1 - Fx_k] <= ubB0
        #             [Fy_k+1 - Fy_k]

        matSelec = numpy.array([ [1, 0],[-1, 1] ])
        footSelec = numpy.array([ [self.f_k_x, 0],[self.f_k_y, 0] ])
        theta_vec = [self.f_k_q,self.F_k_q[0]]

        # rotation matrice from F_k+1 to F_k
        rotMat1 = numpy.array([[cos(theta_vec[0]), sin(theta_vec[0])],[-sin(theta_vec[0]), cos(theta_vec[0])]])
        rotMat2 = numpy.array([[cos(theta_vec[1]), sin(theta_vec[1])],[-sin(theta_vec[1]), cos(theta_vec[1])]])
        nf = self.nf
        nEdges = self.A0l.shape[0]
        N = self.N
        ncfoot = nf * nEdges

        if self.currentSupport.foot == "left":
            A_f1 = self.A0r.dot(rotMat1)
            A_f2 = self.A0l.dot(rotMat2)
            B_f1 = self.ubB0r
            B_f2 = self.ubB0l
        else :
            A_f1 = self.A0l.dot(rotMat1)
            A_f2 = self.A0r.dot(rotMat2)
            B_f1 = self.ubB0l
            B_f2 = self.ubB0r

        tmp1 = numpy.array( [A_f1[:,0],numpy.zeros((nEdges,),dtype=float)] )
        tmp2 = numpy.array( [numpy.zeros((nEdges,),dtype=float),A_f2[:,0]] )
        tmp3 = numpy.array( [A_f1[:,1],numpy.zeros((nEdges,),dtype=float)] )
        tmp4 = numpy.array( [numpy.zeros(nEdges,),A_f2[:,1]] )

        X_mat = numpy.concatenate( (tmp1.T,tmp2.T) , 0)
        A0x = X_mat.dot(matSelec)
        Y_mat = numpy.concatenate( (tmp3.T,tmp4.T) , 0)
        A0y = Y_mat.dot(matSelec)

        B0full = numpy.concatenate( (B_f1, B_f2) , 0 )
        B0 = B0full + X_mat.dot(footSelec[0,:]) + Y_mat.dot(footSelec[1,:])

        self.Afoot[...] = numpy.concatenate ((
            numpy.zeros((ncfoot,N),dtype=float), A0x,
            numpy.zeros((ncfoot,N),dtype=float), A0y
            ), 1
        )
        self.ubBfoot[...] = B0

    def buildFootRotationConstraints(self):
        """ constraints that freeze foot orientation for support leg """
        # 0 = E_F_bar * dF_k_q
        # <=>
        # ( 0 ) = ( E_FR_bar        0 ) * ( Pvs * f_k_qR + Pvu * dddF_k_qR )
        # ( 0 )   (        0 E_FL_bar ) * ( Pvs * f_k_qL + Pvu * dddF_k_qL )
        # <=>
        # 0 = E_FR_bar * Pvs * f_k_qR + E_FR_bar * Pvu * dddF_k_qR
        # 0 = E_FL_bar * Pvs * f_k_qL + E_FL_bar * Pvu * dddF_k_qL
        # <=>
        # E_FR_bar * Pvu * dddF_k_qR = - E_FR_bar * Pvs * f_k_qR
        # E_FL_bar * Pvu * dddF_k_qL = - E_FL_bar * Pvs * f_k_qL
        # <=>
        # A_fvel =
        # (E_FR_bar * Pvu              0 ) * dddF_k_qR
        # (             0 E_FL_bar * Pvu ) * dddF_k_qL
        # B_rot_eq =
        # ( - E_FR_bar * Pvs * f_k_qR)
        # ( - E_FL_bar * Pvs * f_k_qL)

        # rename for convenience
        A_fvel_eq_R = self.A_fvel_eq[:, :self.N]
        A_fvel_eq_L = self.A_fvel_eq[:, self.N:]
        B_fvel_eq   = self.B_fvel_eq

        # calculate proper selection matrices
        self._update_foot_selection_matrices()

        # build foot angular velocity constraints
        # A_fvel_eq =
        # (E_FR_bar * Pvu              0 ) * dddF_k_qR
        # (             0 E_FL_bar * Pvu ) * dddF_k_qL
        # B_fvel_eq =
        # ( - E_FR_bar * Pvs * f_k_qR)
        # ( - E_FL_bar * Pvs * f_k_qL)
        A_fvel_eq_R[...] = self.E_FR_bar.dot(self.Pvu)
        A_fvel_eq_L[...] = self.E_FL_bar.dot(self.Pvu)

        B_fvel_eq[...]   = -self.E_FR_bar.dot(self.Pvs).dot(self.f_k_qR) \
                           -self.E_FL_bar.dot(self.Pvs).dot(self.f_k_qL)

    def buildRotIneqConstraint(self):
        """ constraints on relative angular velocity """
        # rename for convenience
        A_fpos_ineq   = self.A_fpos_ineq
        A_fpos_ineq_R = A_fpos_ineq[:, :self.N]
        A_fpos_ineq_L = A_fpos_ineq[:, self.N:]
        ubB_fpos_ineq = self.ubB_fpos_ineq
        lbB_fpos_ineq = self.lbB_fpos_ineq

        A_fvel_ineq   = self.A_fvel_ineq
        A_fvel_ineq_R = A_fvel_ineq[:, :self.N]
        A_fvel_ineq_L = A_fvel_ineq[:, self.N:]
        ubB_fvel_ineq = self.ubB_fvel_ineq
        lbB_fvel_ineq = self.lbB_fvel_ineq

        # calculate proper selection matrices
        self._update_foot_selection_matrices()

        # build foot position constraints
        # || F_kp1_qR - F_kp1_qL ||_2^2 <= 0.09 ~ 5 degrees
        # <=>
        # -0.09 <= F_kp1_qR - F_kp1_qL <= 0.09
        # -0.09 - Pps(f_k_qR - f_k_qL) <= Ppu * ( 1 | -1 ) U_k <= 0.09 - Pps(f_k_qR - f_k_qL)
        A_fpos_ineq_R[...] =  numpy.eye(self.N)
        A_fpos_ineq_L[...] = -numpy.eye(self.N)
        A_fpos_ineq[...]   = self.Ppu.dot(A_fpos_ineq)

        ubB_fpos_ineq[...] =  0.09 - self.Pps.dot(self.f_k_qR - self.f_k_qL)
        lbB_fpos_ineq[...] = -0.09 - self.Pps.dot(self.f_k_qR - self.f_k_qL)

        # build foot velocity constraints
        A_fvel_ineq_R[...] =  numpy.eye(self.N)
        A_fvel_ineq_L[...] = -numpy.eye(self.N)
        A_fvel_ineq[...]   = self.Pvu.dot(A_fvel_ineq)

        ubB_fvel_ineq[...] =  0.22 - self.Pvs.dot(self.f_k_qR - self.f_k_qL)
        lbB_fvel_ineq[...] = -0.22   - self.Pvs.dot(self.f_k_qR - self.f_k_qL)

    def solve(self):
        """
        Solve problem on given prediction horizon with implemented solver.
        """
        err_str = 'Please derive from this class to implement your problem and solver'
        raise NotImplementedError(err_str)
