import sys
import numpy
import utility
import matplotlib.pyplot as plt

from base_traj import BaseGeneratorTraj
from visualization_traj import PlotData
from walking_generator.utility import color_matrix

# Try to get qpOASES SQP Problem class
try:
    from qpoases import PyOptions as Options
    from qpoases import PyPrintLevel as PrintLevel
    from qpoases import PySQProblem as SQProblem
    from qpoases import PySolutionAnalysis as SolutionAnalysis
except ImportError:
    err_str = 'Please install qpOASES python interface, else you will not be able to use this pattern generator.'
    raise ImportError(err_str)

class NMPCGeneratorTraj(BaseGeneratorTraj):
    """
    Implementation of the combined problems using NMPC techniques.

    Solve QP for position and orientation of CoM and feet simultaneously in
    each timestep. Calculates derivatives and updates states in each step.
    """
    def __init__(
        self, N=16, T=0.2, T_step=1.6,
        fsm_state='D', fsm_sl=1
    ):
        """
        Initialize pattern generator matrices through base class
        and allocate two QPs one for optimzation of orientation and
        one for position of CoM and feet.

        """
        super(NMPCGeneratorTraj, self).__init__(
            N, T, T_step, fsm_state, fsm_sl
        )
        # The pattern generator has to solve the following kind of
        # problem in each iteration

        # min_x 1/2 * x.T * H(w0) * x + x.T g(w0)
        # s.t.   lbA(w0) <= A(w0) * x <= ubA(w0)
        #         lb(w0) <=         x <= ub(wo)

        # Because of varying H and A, we have to use the
        # SQPProblem class, which supports this kind of QPs

        # rename for convenience
        N  = self.N
        nf = self.nf

        # define some qpOASES specific things
        self.cpu_time = numpy.array([0.1]) # upper bound on CPU time, 0 is no upper limit
        self.nwsr     = numpy.array([100])# # of working set recalculations
        self.options = Options()
        self.options.setToMPC()
        self.options.printLevel = PrintLevel.LOW

        # define variable dimensions
        # variables of:     position + orientation
        self.nv = 2*(self.N+self.nf) + 2*N

        # define constraint dimensions
        self.nc_pos = (
              2*self.nc_cop
            + self.nc_foot_position
            + self.nc_fchange_eq
        )
        self.nc_ori = (
              self.nc_fvel_eq
            + self.nc_fpos_ineq
            + self.nc_fvel_ineq
        )
        self.nc = (
            # position
              self.nc_pos
            # orientation
            + self.nc_ori
        )

        # setup problem
        self.dofs = numpy.zeros(self.nv)
        self.qp   = SQProblem(self.nv, self.nc)

        # load NMPC options
        self.qp.setOptions(self.options)

        self.qp_H   =  numpy.eye(self.nv,self.nv)
        self.qp_A   =  numpy.zeros((self.nc,self.nv))
        self.qp_g   =  numpy.zeros((self.nv,))
        self.qp_lb  = -numpy.ones((self.nv,))*1e+08
        self.qp_ub  =  numpy.ones((self.nv,))*1e+08
        self.qp_lbA = -numpy.ones((self.nc,))*1e+08
        self.qp_ubA =  numpy.ones((self.nc,))*1e+08

        self._qp_is_initialized = False

        # save computation time and working set recalculations
        self.qp_nwsr    = numpy.array([0.0])
        self.qp_cputime = numpy.array([0.0])

        # setup analyzer for solution analysis
        analyser = SolutionAnalysis()

        # helper matrices for common expressions
        self.Hx     = numpy.zeros((1, 2*(N+nf)), dtype=float)
        self.Q_k_x  = numpy.zeros((N+nf, N+nf),  dtype=float)
        self.p_k_x  = numpy.zeros((N+nf,),       dtype=float)
        self.p_k_y  = numpy.zeros((N+nf,),       dtype=float)

        self.Hq     = numpy.zeros((1, 2*N), dtype=float)
        self.Q_k_qR = numpy.zeros((N, N),   dtype=float)
        self.Q_k_qL = numpy.zeros((N, N),   dtype=float)
        self.p_k_qR = numpy.zeros((N),      dtype=float)
        self.p_k_qL = numpy.zeros((N),      dtype=float)

        self.A_pos_x   = numpy.zeros((self.nc_pos, 2*(N+nf)), dtype=float)
        self.A_pos_q   = numpy.zeros((self.nc_pos, 2*N), dtype=float)
        self.ubA_pos = numpy.zeros((self.nc_pos,), dtype=float)
        self.lbA_pos = numpy.zeros((self.nc_pos,), dtype=float)

        self.A_ori   = numpy.zeros((self.nc_ori, 2*N), dtype=float)
        self.ubA_ori = numpy.zeros((self.nc_ori,),     dtype=float)
        self.lbA_ori = numpy.zeros((self.nc_ori,),     dtype=float)

        self.derv_Acop_map = numpy.zeros((self.nc_cop, self.N), dtype=float)
        self.derv_Afoot_map = numpy.zeros((self.nc_foot_position, self.N), dtype=float)

        self._update_foot_selection_matrix()

        # add additional keys that should be saved
        self._data_keys.append('qp_nwsr')
        self._data_keys.append('qp_cputime')

        # reinitialize plot data structure
        self.data = PlotData(self)

        self.count = 0
        self.previous_return_value = 0

    def solve(self):
        """ Process and solve problem, s.t. pattern generator data is consistent """
        self._preprocess_solution()
        return_value = self._solve_qp()
        # print("------------------",return_value,"------------------")
        if return_value != 0:
            self.count += 1
        elif self.previous_return_value != 0:
            self.count = 0
        self._postprocess_solution()
        self.previous_return_value = return_value
        # print("***",self.count,"***")
        return self.count

    def _preprocess_solution(self):
        """ Update matrices and get them into the QP data structures """
        # rename for convenience
        N  = self.N
        nf = self.nf

        # dofs
        dofs = self.dofs

        dddC_k_x  = self.dddC_k_x
        F_k_x     = self.F_k_x
        dddC_k_y  = self.dddC_k_y
        F_k_y     = self.F_k_y
        dddF_k_qR = self.dddF_k_qR
        dddF_k_qL = self.dddF_k_qL

        # inject dofs for convenience
        # dofs = ( dddC_k_x ) N
        #        (    F_k_x ) nf
        #        ( dddC_k_y ) N
        #        (    F_k_y ) nf
        #        ( dddF_k_q ) N
        #        ( dddF_k_q ) N
        dofs[0  :0+N   ] = dddC_k_x
        dofs[0+N:0+N+nf] = F_k_x

        a = N+nf
        dofs[a  :a+N   ] = dddC_k_y
        dofs[a+N:a+N+nf] = F_k_y

        a = 2*(N+nf)
        dofs[  a:a+N]    = dddF_k_qR
        dofs[ -N:]       = dddF_k_qL

        # define position and orientation dofs
        # U_k = ( U_k_xy, U_k_q).T
        # U_k_xy = ( dddC_k_x ) N
        #          (    F_k_x ) nf
        #          ( dddC_k_y ) N
        #          (    F_k_y ) nf
        # U_k_q  = ( dddF_k_q ) N
        #          ( dddF_k_q ) N

        # position dofs
        U_k    = self.dofs
        U_k_xy = U_k[    :2*(N+nf)]
        U_k_x  = U_k_xy[:(N+nf)]
        U_k_y  = U_k_xy[(N+nf):]

        # orientation dofs
        U_k_q   = U_k  [-2*N: ]
        U_k_qR  = U_k_q[    :N]
        U_k_qL  = U_k_q[   N: ]

        # position dimensions
        nU_k    = U_k.shape[0]
        nU_k_xy = U_k_xy.shape[0]
        nU_k_x  = U_k_x.shape[0]
        nU_k_y  = U_k_y.shape[0]

        # orientation dimensions
        nU_k_q  = U_k_q.shape[0]
        nU_k_qR = U_k_qR.shape[0]
        nU_k_qL = U_k_qL.shape[0]

        # initialize with actual values, else take last known solution
        # NOTE for warmstart last solution is taken from qpOASES internal memory
        if not self._qp_is_initialized:
            # TODO guess initial active set
            # this requires changes to the python interface
            pass

        # calculate some common sub expressions
        self._calculate_common_expressions()

        # calculate Jacobian parts that are non-trivial, i.e. wrt. to orientation
        self._calculate_derivatives()

        # POSITION QP
        # rename matrices
        Q_k_x = self.Q_k_x
        Q_k_y = self.Q_k_x # NOTE it's exactly the same!
        p_k_x = self.p_k_x
        p_k_y = self.p_k_y

        # ORIENTATION QP
        # rename matrices
        Q_k_qR = self.Q_k_qR
        Q_k_qL = self.Q_k_qL
        p_k_qR = self.p_k_qR
        p_k_qL = self.p_k_qL

        # define QP matrices
        # Gauss-Newton Hessian approximation
        # define sub blocks
        # H = (Hxy | Hqx)
        #     (Hxq | Hqq)
        Hxx = self.qp_H[:nU_k_xy,:nU_k_xy]
        Hxq = self.qp_H[:nU_k_xy,nU_k_xy:]
        Hqx = self.qp_H[nU_k_xy:,:nU_k_xy]
        Hqq = self.qp_H[nU_k_xy:,nU_k_xy:]

        # fill matrices
        Hxx[ :nU_k_x, :nU_k_x] = Q_k_x
        Hxx[-nU_k_y:,-nU_k_y:] = Q_k_y

        Hqq[ :nU_k_qR, :nU_k_qR] = Q_k_qR
        Hqq[-nU_k_qL:,-nU_k_qL:] = Q_k_qL
        #self.qp_H[...] = numpy.eye(self.nv)

        # Gradient of Objective
        # define sub blocks
        # g = (gx)
        #     (gq)
        gx = self.qp_g[:nU_k_xy]
        gq = self.qp_g[-nU_k_q:]

        # gx = ( U_k_x.T Q_k_x + p_k_x )
        gx[ :nU_k_x] = U_k_x.dot(Q_k_x) + p_k_x
        gx[-nU_k_y:] = U_k_y.dot(Q_k_y) + p_k_y

        # gq = ( U_k_q.T Q_k_q + p_k_q )
        gq[ :nU_k_qR] = U_k_qR.dot(Q_k_qR) + p_k_qR
        gq[-nU_k_qL:] = U_k_qL.dot(Q_k_qL) + p_k_qL

        # CONSTRAINTS
        # A = ( A_xy, A_xyq )
        #     (    0, A_q   )
        A_xy   = self.qp_A  [:self.nc_pos,:nU_k_xy]
        A_xyq  = self.qp_A  [:self.nc_pos,-nU_k_q:]
        lbA_xy = self.qp_lbA[:self.nc_pos]
        ubA_xy = self.qp_ubA[:self.nc_pos]

        A_q   = self.qp_A  [-self.nc_ori:,-nU_k_q:]
        lbA_q = self.qp_lbA[-self.nc_ori:]
        ubA_q = self.qp_ubA[-self.nc_ori:]

        # linearized constraints are given by
        # lbA - A * U_k <= nablaA * Delta_U_k <= ubA - A * U_k
        A_xy[...]   = self.A_pos_x
        A_xyq[...]  = self.A_pos_q
        lbA_xy[...] = self.lbA_pos - self.A_pos_x.dot(U_k_xy)
        ubA_xy[...] = self.ubA_pos - self.A_pos_x.dot(U_k_xy)

        A_q[...]   = self.A_ori
        lbA_q[...] = self.lbA_ori - self.A_ori.dot(U_k_q)
        ubA_q[...] = self.ubA_ori - self.A_ori.dot(U_k_q)

    def _calculate_common_expressions(self):
        """
        encapsulation of complicated matrix assembly of former orientation and
        position QP sub matrices
        """
        #rename for convenience
        N  = self.N
        nf = self.nf

        # weights
        alpha = self.a
        beta  = self.b
        gamma = self.c
        delta = self.d

        # matrices
        E_FR = self.E_FR
        E_FL = self.E_FL

        Pps = self.Pps
        Ppu = self.Ppu
        Pzs = self.Pzs
        Pzu = self.Pzu

        c_k_x = self.c_k_x
        c_k_y = self.c_k_y
        f_k_x = self.f_k_x
        f_k_y = self.f_k_y
        f_k_qR = self.f_k_qR
        f_k_qL = self.f_k_qL

        v_kp1 = self.v_kp1
        V_kp1 = self.V_kp1

        # print(v_kp1)
        # print(V_kp1)
        # print(f_k_x,f_k_y)

        C_kp1_x_ref = self.C_kp1_x_ref
        C_kp1_y_ref = self.C_kp1_y_ref
        C_kp1_q_ref = self.C_kp1_q_ref

        # POSITION QP MATRICES
        # Q_k_x = ( Q_k_xXX Q_k_xXF ) = Q_k_y
        #         ( Q_k_xFX Q_k_xFF )
        Q_k_x = self.Q_k_x

        a = 0; b = N
        c = 0; d = N
        Q_k_xXX = Q_k_x[a:b,c:d]

        a = 0; b = N
        c = N; d = N+nf
        Q_k_xXF = Q_k_x[a:b,c:d]

        a = N; b = N+nf
        c = 0; d = N
        Q_k_xFX = Q_k_x[a:b,c:d]

        a = N; b = N+nf
        c = N; d = N+nf
        Q_k_xFF = Q_k_x[a:b,c:d]

        # # Q_k_xXX = (  0.5 * a * Pvu^T * Pvu + c * Pzu^T * Pzu + d * I )
        # Q_k_xXF = ( -0.5 * c * Pzu^T * V_kp1 )
        # Q_k_xFX = ( -0.5 * c * Pzu^T * V_kp1 )^T
        # Q_k_xFF = (  0.5 * c * V_kp1^T * V_kp1 )
        Q_k_xXX[...] = (
              alpha * Ppu.transpose().dot(Ppu)
            + gamma * Pzu.transpose().dot(Pzu)
            + delta * numpy.eye(N)
        )
        Q_k_xXF[...] = - gamma * Pzu.transpose().dot(V_kp1)
        Q_k_xFX[...] = Q_k_xXF.transpose()
        Q_k_xFF[...] =   gamma * V_kp1.transpose().dot(V_kp1)

        # p_k_x = ( p_k_xX )
        #         ( p_k_xF )
        p_k_x = self.p_k_x
        p_k_xX = p_k_x[   :N]
        p_k_xF = p_k_x[-nf: ]

        # # p_k_xX = (  0.5 * a * Pvu^T * Pvu + c * Pzu^T * Pzu + d * I )
        # # p_k_xF = ( -0.5 * c * Pzu^T * V_kp1 )
        p_k_xX[...] = alpha * Ppu.transpose().dot(Pps.dot(c_k_x) - C_kp1_x_ref) \
                    + gamma * Pzu.transpose().dot(Pzs.dot(c_k_x) - v_kp1.dot(f_k_x))
        p_k_xF[...] =-gamma * V_kp1.transpose().dot(Pzs.dot(c_k_x) - v_kp1.dot(f_k_x))

        # p_k_y = ( p_k_yX )
        #         ( p_k_yF )
        p_k_y = self.p_k_y
        p_k_yX = p_k_y[   :N]
        p_k_yF = p_k_y[-nf: ]

        # # p_k_yX = (  0.5 * a * Pvu^T * Pvu + c * Pzu^T * Pzu + d * I )
        # # p_k_yF = ( -0.5 * c * Pzu^T * V_kp1 )
        p_k_yX[...] = alpha * Ppu.transpose().dot(  Pps.dot(c_k_y) - C_kp1_y_ref) \
                    + gamma * Pzu.transpose().dot(  Pzs.dot(c_k_y) - v_kp1.dot(f_k_y))
        p_k_yF[...] =-gamma * V_kp1.transpose().dot(Pzs.dot(c_k_y) - v_kp1.dot(f_k_y))

        # ORIENTATION QP MATRICES
        # # Q_k_qR = ( 0.5 * a * Pvu^T * E_FR^T *  E_FR * Pvu )
        Q_k_qR = self.Q_k_qR
        Q_k_qR[...] = alpha * Ppu.transpose().dot(E_FR.transpose()).dot(E_FR).dot(Ppu)

        # # p_k_qR = (       a * Pvu^T * E_FR^T * (E_FR * Pvs * f_k_qR + dC_kp1_q_ref) )
        p_k_qR = self.p_k_qR
        p_k_qR[...] = alpha * Ppu.transpose().dot(E_FR.transpose()).dot(E_FR.dot(Pps).dot(f_k_qR) - C_kp1_q_ref)

        # # Q_k_qL = ( 0.5 * a * Pvu^T * E_FL^T *  E_FL * Pvu )
        Q_k_qL = self.Q_k_qL
        Q_k_qL[...] = alpha * Ppu.transpose().dot(E_FL.transpose()).dot(E_FL).dot(Ppu)
        # # p_k_qL = (       a * Pvu^T * E_FL^T * (E_FL * Pvs * f_k_qL + dC_kp1_q_ref) )
        p_k_qL = self.p_k_qL
        p_k_qL[...] = alpha * Ppu.transpose().dot(E_FL.transpose()).dot(E_FL.dot(Pps).dot(f_k_qL) - C_kp1_q_ref)

        # LINEAR CONSTRAINTS
        # CoP constraints
        a = 0
        b = self.nc_cop
        self.A_pos_x[a:b] = self.Acop
        self.lbA_pos[a:b] = self.lbBcop
        self.ubA_pos[a:b] = self.ubBcop

        # CP terminal constraint
        a = self.nc_cop
        b = self.nc_cop + self.nc_dcm
        self.A_pos_x[a:b] = self.Adcm
        self.lbA_pos[a:b] = self.lbBdcm
        self.ubA_pos[a:b] = self.ubBdcm

        #foot inequality constraints
        a = self.nc_cop + self.nc_dcm
        b = self.nc_cop + self.nc_dcm + self.nc_foot_position
        self.A_pos_x[a:b] = self.Afoot
        self.lbA_pos[a:b] = self.lbBfoot
        self.ubA_pos[a:b] = self.ubBfoot

        #foot equality constraints
        a = self.nc_cop + self.nc_dcm + self.nc_foot_position
        b = self.nc_cop + self.nc_dcm + self.nc_foot_position + self.nc_fchange_eq
        self.A_pos_x[a:b] = self.eqAfoot
        self.lbA_pos[a:b] = self.eqBfoot
        self.ubA_pos[a:b] = self.eqBfoot

        # velocity constraints on support foot to freeze movement
        a = 0
        b = self.nc_fvel_eq
        self.A_ori  [a:b] = self.A_fvel_eq
        self.lbA_ori[a:b] = self.B_fvel_eq
        self.ubA_ori[a:b] = self.B_fvel_eq

        # box constraints for maximum orientation change
        a = self.nc_fvel_eq
        b = self.nc_fvel_eq + self.nc_fpos_ineq
        self.A_ori  [a:b] = self.A_fpos_ineq
        self.lbA_ori[a:b] = self.lbB_fpos_ineq
        self.ubA_ori[a:b] = self.ubB_fpos_ineq

        # box constraints for maximum angular velocity
        a = self.nc_fvel_eq + self.nc_fpos_ineq
        b = self.nc_fvel_eq + self.nc_fpos_ineq + self.nc_fvel_ineq
        self.A_ori  [a:b] = self.A_fvel_ineq
        self.lbA_ori[a:b] = self.lbB_fvel_ineq
        self.ubA_ori[a:b] = self.ubB_fvel_ineq

    def _calculate_derivatives(self):
        """ calculate the Jacobian of the constraints function """

        # COP CONSTRAINTS
        # build the constraint enforcing the center of pressure to stay inside
        # the support polygon given through the convex hull of the foot.

        # define dummy values
        # D_kp1 = (D_kp1x, Dkp1_y)
        D_kp1  = numpy.zeros( (self.nFootEdge*self.N, 2*self.N), dtype=float )
        D_kp1x = D_kp1[:, :self.N] # view on big matrix
        D_kp1y = D_kp1[:,-self.N:] # view on big matrix
        b_kp1 = numpy.zeros( (self.nFootEdge*self.N,), dtype=float )

        # change entries according to support order changes in D_kp1
        theta_vec = [self.f_k_q,self.F_k_q[0],self.F_k_q[1]]
        for i in range(self.N):
            theta = theta_vec[self.supportDeque[i].stepNumber]

            # NOTE THIS CHANGES DUE TO APPLYING THE DERIVATIVE!
            rotMat = numpy.array([
                # old
                # [ cos(theta), sin(theta)],
                # [-sin(theta), cos(theta)]
                # new: derivative wrt to theta
                [-numpy.sin(theta), numpy.cos(theta)],
                [-numpy.cos(theta),-numpy.sin(theta)]
            ])

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
                D_kp1x[i*self.nFootEdge+k, i] = A0[k][0]
                # get d_i+1^y(f^theta)
                D_kp1y[i*self.nFootEdge+k, i] = A0[k][1]
                # get right hand side of equation
                b_kp1 [i*self.nFootEdge+k]    = B0[k]

        #rename for convenience
        N  = self.N
        nf = self.nf
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
        # NOTE D_kp1 is member and D_kp1 = ( D_kp1x | D_kp1y )
        #      D_kp1x,y contains entries from support polygon
        dummy = D_kp1.dot(PzuV)
        dummy = dummy.dot(self.dofs[:2*(N+nf)])

        # CoP constraints
        a = 0
        b = self.nc_cop
        self.A_pos_q[a:b, :N] = \
           dummy.dot(self.derv_Acop_map).dot(self.E_FR_bar).dot(self.Ppu)

        self.A_pos_q[a:b,-N:] = \
            dummy.dot(self.derv_Acop_map).dot(self.E_FL_bar).dot(self.Ppu)

        # FOOT POSITION CONSTRAINTS
        # defined on the horizon
        # inequality constraint on both feet A u + B <= 0
        # A0 R(theta) [Fx_k+1 - Fx_k] <= ubB0
        #             [Fy_k+1 - Fy_k]

        matSelec = numpy.array([ [1, 0],[-1, 1] ])
        footSelec = numpy.array([ [self.f_k_x, 0],[self.f_k_y, 0] ])
        theta_vec = [self.f_k_q, self.F_k_q[0]]

        # rotation matrice from F_k+1 to F_k
        # NOTE THIS CHANGES DUE TO APPLYING THE DERIVATIVE!
        rotMat1 = numpy.array([
            # old
            # [cos(theta_vec[0]), sin(theta_vec[0])],
            # [-sin(theta_vec[0]), cos(theta_vec[0])]
            # new: derivative wrt to theta
            [-numpy.sin(theta_vec[0]), numpy.cos(theta_vec[0])],
            [-numpy.cos(theta_vec[0]),-numpy.sin(theta_vec[0])]
        ])
        rotMat2 = numpy.array([
            # old
            # [cos(theta_vec[1]), sin(theta_vec[1])],
            # [-sin(theta_vec[1]), cos(theta_vec[1])]
            # new
            [-numpy.sin(theta_vec[1]), numpy.cos(theta_vec[1])],
            [-numpy.cos(theta_vec[1]),-numpy.sin(theta_vec[1])]
        ])
        nf = self.nf
        nEdges = self.A0l.shape[0]
        N = self.N
        ncfoot = nf * nEdges

        if self.currentSupport.foot == "left":
            A_f1 = self.A0r.dot(rotMat1)
            A_f2 = self.A0l.dot(rotMat2)
        else :
            A_f1 = self.A0l.dot(rotMat1)
            A_f2 = self.A0r.dot(rotMat2)

        tmp1 = numpy.array( [A_f1[:,0],numpy.zeros((nEdges,),dtype=float)] )
        tmp2 = numpy.array( [numpy.zeros((nEdges,),dtype=float),A_f2[:,0]] )
        tmp3 = numpy.array( [A_f1[:,1],numpy.zeros((nEdges,),dtype=float)] )
        tmp4 = numpy.array( [numpy.zeros(nEdges,),A_f2[:,1]] )

        X_mat = numpy.concatenate( (tmp1.T,tmp2.T) , 0)
        A0x = X_mat.dot(matSelec)
        Y_mat = numpy.concatenate( (tmp3.T,tmp4.T) , 0)
        A0y = Y_mat.dot(matSelec)

        dummy = numpy.concatenate ((
            numpy.zeros((ncfoot,N),dtype=float), A0x,
            numpy.zeros((ncfoot,N),dtype=float), A0y
            ), 1
        )
        dummy = dummy.dot(self.dofs[:2*(N+nf)])

        #foot inequality constraints
        a = self.nc_cop
        b = self.nc_cop + self.nc_foot_position
        self.A_pos_q[a:b, :N] = dummy.dot(self.derv_Afoot_map).dot(self.E_FR_bar).dot(self.Ppu)
        self.A_pos_q[a:b,-N:] = dummy.dot(self.derv_Afoot_map).dot(self.E_FL_bar).dot(self.Ppu)

    def _solve_qp(self):
        """
        Solve QP first run with init functionality and other runs with warmstart
        """
        self.cpu_time = 2.9 # ms
        self.nwsr = 1000 # unlimited bounded

        if not self._qp_is_initialized:
            return_value = self.qp.init(
                self.qp_H, self.qp_g, self.qp_A,
                self.qp_lb, self.qp_ub,
                self.qp_lbA, self.qp_ubA,
                self.nwsr, self.cpu_time
            )
            nwsr, cputime = self.nwsr, self.cpu_time
            self._qp_is_initialized = True
        else:
                return_value = self.qp.hotstart(
                    self.qp_H, self.qp_g, self.qp_A,
                    self.qp_lb, self.qp_ub,
                    self.qp_lbA, self.qp_ubA,
                    self.nwsr, self.cpu_time
                )
                nwsr, cputime = self.nwsr, self.cpu_time
                if return_value != 0:
                    print("--- Error ! ---")

        # orientation primal solution
        self.qp.getPrimalSolution(self.dofs)

        # save qp solver data
        self.qp_nwsr    = nwsr          # working set recalculations
        self.qp_cputime = cputime*1000. # in milliseconds (set to 2.9ms)

        return return_value

    def _postprocess_solution(self):
        """ Get solution and put it back into generator data structures """
        # rename for convenience
        N  = self.N
        nf = self.nf

        # extract dofs
        # dofs = ( dddC_k_x ) N
        #        (    F_k_x ) nf
        #        ( dddC_k_y ) N
        #        (    F_k_y ) nf
        #        ( dddF_k_q ) N
        #        ( dddF_k_q ) N

        # NOTE this time we add an increment to the existing values
        # data(k+1) = data(k) + alpha * dofs

        # TODO add line search when problematic
        alpha = 1.0

        # x values
        self.dddC_k_x[:]  += alpha * self.dofs[0  :0+N   ]
        self.F_k_x[:]     += alpha * self.dofs[0+N:0+N+nf]

        # print(self.F_k_x)

        # y values
        a = N + nf
        self.dddC_k_y[:]  += alpha * self.dofs[a  :a+N   ]
        self.F_k_y[:]     += alpha * self.dofs[a+N:a+N+nf]

        # feet orientation
        a =2*(N + nf)
        self.dddF_k_qR[:] += alpha * self.dofs[  a:a+N]
        self.dddF_k_qL[:] += alpha * self.dofs[ -N:]

    def update(self):
        """
        overload update function to define time dependent support foot selection
        matrix.
        """
        ret = super(NMPCGeneratorTraj, self).update()

        # update selection matrix when something has changed
        self._update_foot_selection_matrix()

        return ret

    def _update_foot_selection_matrix(self):
        """ get right foot selection matrix """
        i = 0
        for j in range(self.N):
            self.derv_Acop_map[i:i+self.nFootEdge,j] = 1.0
            i += self.nFootEdge

        self.derv_Afoot_map[...] = 0.0

        i = self.nFootPosHullEdges
        for j in range(self.nf-1):
            for k in range(self.N):
                if self.V_kp1[k,j] == 1:
                    self.derv_Afoot_map[i:i+self.nFootPosHullEdges,k] = 1.0
                    i += self.nFootPosHullEdges
                    break
            else:
                self.derv_Afoot_map[i:i+self.nFootPosHullEdges,j] = 0.0
