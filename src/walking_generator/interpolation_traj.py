import numpy
from copy import deepcopy
from helper import CoMState, ZMPState, BaseTypeFoot
from base_traj import BaseGeneratorTraj
import math

class Interpolation(object):
    """
    Interpolation class of walking pattern generator for humanoids, cf.
    LAAS-UHEI walking report.

    Interpolation Class provides all methods to interpolated from the solution of the
    pattern generator. It interpolate the CoM, the ZMP and the Feet state along the
    whole trajectory with a given interpolation period (input)
    """
    def __init__(self, Tc=0.001, BG=BaseGeneratorTraj() ):

        # the generator is supposed to have been initialized before
        self.gen = BG

        self.T = self.gen.T # QP sampling period
        self.Tc = Tc # sampling period of the robot low level controller
        self.interval = int(self.T/self.Tc) # number of iteration in 100ms
                                            # the initial state of the next QP iteration

        # initiale states used to interpolate (they should be intialized once at
        # the beginning of the qp
        # and updated inside the class
        self.curCoM = CoMState()
        self.curCoM.x = deepcopy(self.gen.c_k_x)
        self.curCoM.y = deepcopy(self.gen.c_k_y)
        self.curCoM.theta = deepcopy(self.gen.c_k_q)
        self.curCoM.h_com = deepcopy(self.gen.h_com)
        zmp = ZMPState()
        zmp.x = self.curCoM.x[0] - self.curCoM.h_com / self.gen.g * self.curCoM.x[2]
        zmp.y = self.curCoM.y[0] - self.curCoM.h_com / self.gen.g * self.curCoM.y[2]

        self.fi = FootInterpolation()
        self.curleft = BaseTypeFoot()
        self.curRight = BaseTypeFoot()
        if self.gen.currentSupport.foot == "left" :
            self.curleft.x = deepcopy(self.gen.f_k_x)
            self.curleft.y = deepcopy(self.gen.f_k_y)
            self.curleft.q = deepcopy(self.gen.f_k_q)
            self.curRight.x = deepcopy(self.gen.f_k_x + self.fi.feetDist * math.sin(self.gen.f_k_q) )
            self.curRight.y = deepcopy(self.gen.f_k_y - self.fi.feetDist * math.cos(self.gen.f_k_q) )
            self.curRight.q = deepcopy(self.gen.f_k_q)
        else :
            self.curleft.x = deepcopy(self.gen.f_k_x)
            self.curleft.y = deepcopy(self.gen.f_k_y)
            self.curleft.q = deepcopy(self.gen.f_k_q)
            self.curRight.x = deepcopy(self.gen.f_k_x - self.fi.feetDist * math.sin(self.gen.f_k_q) )
            self.curRight.y = deepcopy(self.gen.f_k_y + self.fi.feetDist * math.cos(self.gen.f_k_q) )
            self.curRight.q = deepcopy(self.gen.f_k_q)

        self.CoMbuffer = numpy.empty( (self.interval,) , dtype=CoMState ) #buffer containing the CoM trajectory over 100ms
        self.ZMPbuffer = numpy.empty( (self.interval,) , dtype=ZMPState ) #buffer containing the ZMP trajectory over 100ms
        self.RFbuffer = numpy.empty( (self.interval,) , dtype=BaseTypeFoot ) #buffer containing the rigth foot trajectory over 100ms
        self.LFbuffer = numpy.empty( (self.interval,) , dtype=BaseTypeFoot ) #buffer containing the left foot trajectory over 100ms

        for i in range(self.interval):
            self.CoMbuffer[i] = deepcopy(self.curCoM)
            self.ZMPbuffer[i] = deepcopy(zmp)
            self.RFbuffer[i] = deepcopy(self.curRight)
            self.LFbuffer[i] = deepcopy(self.curleft)

        self.comTraj = [] #buffer containing the full CoM trajectory
        self.zmpTraj = [] #buffer containing the full ZMP trajectory
        self.leftFootTraj = [] #buffer containing the full left foot trajectory
        self.rightFootTraj = [] #buffer containing the full right foot trajectory

        for i in range(30):
            self.comTraj.extend(deepcopy(self.CoMbuffer))
            self.zmpTraj.extend(deepcopy(self.ZMPbuffer))
            self.leftFootTraj .extend(deepcopy(self.LFbuffer))
            self.rightFootTraj.extend(deepcopy(self.RFbuffer))

        self.lipm = LIPM(self.T,self.Tc,self.curCoM.h_com)
        self.fi = FootInterpolation(genrator=self.gen)

    def interpolate(self, time):

        self.curCoM, self.CoMbuffer, self.ZMPbuffer = self.lipm.interpolate(
                                self.gen.dddC_k_x[0], self.gen.dddC_k_y[0],
                                self.curCoM, self.ZMPbuffer, self.CoMbuffer)
        self.curleft, self.curRight, self.LFbuffer, self.RFbuffer =\
                        self.fi.interpolate(time, self.gen.currentSupport,
                                    self.curleft, self.curRight,
                                    self.gen.F_k_x[0], self.gen.F_k_y[0], self.gen.F_k_q[0],
                                    self.LFbuffer, self.RFbuffer)

        for i in range ( self.LFbuffer.shape[0] ) :
            self.CoMbuffer[i].q[0] = 0.5*(self.LFbuffer[i].q + self.RFbuffer[i].q)
            self.CoMbuffer[i].q[1] = 0.5*(self.LFbuffer[i].dq + self.RFbuffer[i].dq)
            self.CoMbuffer[i].q[2] = 0.5*(self.LFbuffer[i].ddq + self.RFbuffer[i].ddq)


        self.comTraj.extend(deepcopy(self.CoMbuffer))
        self.zmpTraj.extend(deepcopy(self.ZMPbuffer))
        self.leftFootTraj .extend(deepcopy(self.LFbuffer))
        self.rightFootTraj.extend(deepcopy(self.RFbuffer))

    def save_to_file(self,filename):
        comX   = numpy.asarray([item.x for item in self.comTraj])
        comY   = numpy.asarray([item.y for item in self.comTraj])
        comQ   = numpy.asarray([item.q for item in self.comTraj])
        zmpX   = numpy.asarray([item.x for item in self.zmpTraj])
        zmpY   = numpy.asarray([item.y for item in self.zmpTraj])
        zmpZ   = numpy.asarray([item.z for item in self.zmpTraj])
        rfX    = numpy.asarray([item.  x for item in self.rightFootTraj])
        rfY    = numpy.asarray([item.  y for item in self.rightFootTraj])
        rfZ    = numpy.asarray([item.  z + 0.105 for item in self.rightFootTraj])
        rfQ    = numpy.asarray([item.  q for item in self.rightFootTraj])
        lfX    = numpy.asarray([item.  x for item in self.leftFootTraj])
        lfY    = numpy.asarray([item.  y for item in self.leftFootTraj])
        lfZ    = numpy.asarray([item.  z + 0.105 for item in self.leftFootTraj])
        lfQ    = numpy.asarray([item.  q for item in self.leftFootTraj])

        rfdX    = numpy.asarray([item. dx for item in self.rightFootTraj])
        rfdY    = numpy.asarray([item. dy for item in self.rightFootTraj])
        rfdZ    = numpy.asarray([item. dz for item in self.rightFootTraj])
        rfdQ    = numpy.asarray([item. dq for item in self.rightFootTraj])
        lfdX    = numpy.asarray([item. dx for item in self.leftFootTraj])
        lfdY    = numpy.asarray([item. dy for item in self.leftFootTraj])
        lfdZ    = numpy.asarray([item. dz for item in self.leftFootTraj])
        lfdQ    = numpy.asarray([item. dq for item in self.leftFootTraj])
        rfddX    = numpy.asarray([item.ddx for item in self.rightFootTraj])
        rfddY    = numpy.asarray([item.ddy for item in self.rightFootTraj])
        rfddZ    = numpy.asarray([item.ddz for item in self.rightFootTraj])
        rfddQ    = numpy.asarray([item.ddq for item in self.rightFootTraj])
        lfddX    = numpy.asarray([item.ddx for item in self.leftFootTraj])
        lfddY    = numpy.asarray([item.ddy for item in self.leftFootTraj])
        lfddZ    = numpy.asarray([item.ddz for item in self.leftFootTraj])
        lfddQ    = numpy.asarray([item.ddq for item in self.leftFootTraj])

        lst = [
        comX[:,0], # 1
        comX[:,1], # 2
        comX[:,2], # 3
        comY[:,0], # 4
        comY[:,1], # 5
        comY[:,2], # 6
        comQ[:,0], # 7
        comQ[:,1], # 8
        comQ[:,2], # 9
        zmpX,      # 10
        zmpY,      # 11
        zmpZ,      # 12
        rfX,       # 13
        rfY,       # 14
        rfZ,       # 15
        rfQ,       # 16
        lfX,       # 17
        lfY,       # 18
        lfZ,       # 19
        lfQ]       # 20
#        rfdX,      # 21
#        rfdY ,     # 22
#        rfdZ  ,    # 23
#        rfdQ  ,    # 24
#        lfdX  ,    # 25
#        lfdY  ,    # 26
#        lfdZ  ,    # 27
#        lfdQ  ,    # 28
#        rfddX ,    # 29
#        rfddY ,    # 30
#        rfddZ ,    # 31
#        rfddQ ,    # 32
#        lfddX ,    # 33
#        lfddY ,    # 34
#        lfddZ ,    # 35
#        lfddQ ]    # 36

        data = numpy.asarray(lst).transpose()
        numpy.savetxt(filename, data, delimiter="   ")

class LIPM(object):
    """
    LIPM class of walking pattern generator for humanoids, cf.
    LAAS-UHEI walking report.

    LIPM class provides all methods to interpolate from the solution
    of the pattern generator the CoM and the ZMP according
    to the linearized inverted pendulum (Kajita2003, Walking)
    """

    # define some constants
    g = 9.81

    def __init__(self, controlPeriod=0.2, commandPeriod=0.001, h_com=8.92675352e-01):
        self.Tc=controlPeriod
        self.T=commandPeriod
        self.h_com=h_com
        # for T sampling interpolation
        self.A = numpy.zeros( (3,3) , dtype=float )
        self.B = numpy.zeros( (3,) , dtype=float )
        self.C = numpy.zeros( (3,) , dtype=float )

        # for Tc sampling interpolation
        self.Ac = numpy.zeros( (3,3) , dtype=float )
        self.Bc = numpy.zeros( (3,) , dtype=float )

        self.intervaleSize = int(self.Tc/self.T)

        self.initializeSystem()

    def initializeSystem(self):
        T = self.T
        A = self.A
        A[0][0] = 1 ; A[0][1] = T ; A[0][2] = T*T*0.5 ;
        A[1][1] = 1 ; A[1][2] = T ;
        A[2][2] = 1 ;

        B = self.B
        B[0]= T*T*T/6
        B[1]= T*T/2
        B[2]= T

        C = self.C
        C[0]= 1
        C[1]= 0
        C[2]= -self.h_com/self.g

        Tc = self.Tc
        Ac = self.Ac
        Ac[0][0] = 1 ; Ac[0][1] = Tc ; Ac[0][2] = Tc*Tc*0.5 ;
        Ac[1][1] = 1 ; Ac[1][2] = Tc ;
        Ac[2][2] = 1 ;

        Bc = self.Bc
        Bc[0]= Tc*Tc*Tc/6
        Bc[1]= Tc*Tc/2
        Bc[2]= Tc

    def interpolate(self, jerkX, jerkY, curCoM, ZMPbuffer, CoMbuffer):
        A = self.A
        B = self.B
        C = self.C
        Ac = self.Ac
        Bc = self.Bc


        CoMbuffer = numpy.resize(CoMbuffer,(self.intervaleSize,))
        ZMPbuffer = numpy.resize(ZMPbuffer,(self.intervaleSize,))

        for i in range(self.intervaleSize):
            CoMbuffer[i] = CoMState()
            ZMPbuffer[i] = ZMPState()

        CoMbuffer[0].x[...] = curCoM.x
        CoMbuffer[0].y[...] = curCoM.y
        ZMPbuffer[0].x = C.dot(curCoM.x)
        ZMPbuffer[0].y = C.dot(curCoM.y)

        for i in range(1,self.intervaleSize):
            CoMbuffer[i].x = A.dot(CoMbuffer[i-1].x) + B.dot(jerkX)
            CoMbuffer[i].y = A.dot(CoMbuffer[i-1].y) + B.dot(jerkY)
            ZMPbuffer[i].x = C.dot(CoMbuffer[i].x)
            ZMPbuffer[i].y = C.dot(CoMbuffer[i].y)

        curCoM.x = Ac.dot(curCoM.x) + Bc.dot(jerkX)
        curCoM.y = Ac.dot(curCoM.y) + Bc.dot(jerkY)

        return curCoM, CoMbuffer, ZMPbuffer

class FootInterpolation(object):
    """
    footInterpolation class of walking pattern generator for humanoids, cf.
    LAAS-UHEI walking report.

    footInterpolation class provides all methods to interpolate from the solution
    of the pattern generator. It interpolate the feet trajectory during the QP period
    """

    def __init__(self, genrator=BaseGeneratorTraj(), QPsamplingPeriod=0.2, NbSamplingPreviewed=16, commandPeriod=0.001,
        FeetDistance=0.19, StepHeight=0.05, stepTime=1.6, doubleSupportTime=0.2):

        self.T = QPsamplingPeriod # QP sampling period
        self.Tc = commandPeriod # Low level control period
        self.feetDist = FeetDistance # Normal Distance between both feet
        self.stepHeigth = StepHeight # Standard maximal step height
        self.polynomeX     = Polynome5() # order 5 polynome for continuity
        self.polynomeY     = Polynome5() #  in position, velocity
        self.polynomeQ     = Polynome5() #  and acceleration
        self.polynomeZ     = Polynome4() # order 5 for a defined middle point
        self.TSS = stepTime - doubleSupportTime # Time of single support
        self.TDS = doubleSupportTime # Time of double support
        self.stepTime = stepTime
        self.intervaleSize = int(self.T/self.Tc) # nuber of interpolated sample
        self.gen = genrator
        self.polynomeZ.setParameters(self.TSS,self.stepHeigth,0.0,0.0)
    '''
    Update the currentState to be valide at the next iteration
    and fill up the queue containing all the intermediate values
    '''
    def interpolate(self, time, currentSupport,
        curLeft, curRight,
        F_k_x, F_k_y,PreviewAngle,
        LeftFootBuffer, RightFootBuffer):

        '''
        |---------------||-------|-----------------------------|-------|
        |     DSP*      ||             single support phase            |
        |               ||take   |      flying                 |landing|
        |               || off   |       phase                 |       |

        * DSP : Double Support Phase
        '''
        timelimit = time + numpy.sum(self.gen.v_kp1) * self.T
        for i in range(self.intervaleSize):
            LeftFootBuffer[i] = BaseTypeFoot()
            RightFootBuffer[i] = BaseTypeFoot()

        # print ("t_lim:",timelimit)
        # print ("v:",self.gen.v_kp1)
        # print ("V:",self.gen.V_kp1)
        # print ("dt:",timelimit - self.stepTime)
        # print ("step time",self.stepTime)
        epsilon = 0.02
        # in case of double support the policy is to stay still
        if time + epsilon < timelimit - self.stepTime + self.T :
            # print "double support"
            for i in range(self.intervaleSize):
                LeftFootBuffer[i] = deepcopy(curLeft)
                RightFootBuffer[i] = deepcopy(curRight)
            # we define the z trajectory in the double support phase
            # to allow the robot to take off and land
            # during the whole single support duration
            self.polynomeZ.setParameters(self.TSS,self.stepHeigth,curRight.z,curRight.dz)

            return curLeft,curRight,LeftFootBuffer, RightFootBuffer

        elif time + epsilon > timelimit - self.stepTime + self.T :
            # print "single support"
            # Deal with the lift off time and the landing time. During those period
            # the foot do not move along the x and y axis.

            # this is counted from the last double support phase
            localInterpolationStartTime = time - (timelimit - self.TSS)
            # this coeffincient indicates how long you allow the foot
            # to take off AND land (in %)
            moduleSupportCoefficient = 0.9
            # this time indicates how long the foot will move in x, y and theta
            UnlockedSwingPeriod = self.TSS * moduleSupportCoefficient
            # this time indicates the time limit where the foot should have reach
            # lift off enough to move in x, y and theta
            endOfLiftoff = 0.5 * (self.TSS-UnlockedSwingPeriod)
            # This time show the time where the foot has flight in the air
            SwingTimePassed = 0.0
            if localInterpolationStartTime > endOfLiftoff:
                SwingTimePassed = localInterpolationStartTime - endOfLiftoff
            # This time is the time remaining before the landing
            timeInterval = UnlockedSwingPeriod - SwingTimePassed
            # this time indicates the time limit where the foot should have reach its goal
            # and needs to land
            startLanding = endOfLiftoff + UnlockedSwingPeriod

            # Set the polynomes
            if (currentSupport.foot=="left") :
                supportFoot = curLeft
                flyingFoot  = curRight
            else :
                supportFoot = curRight
                flyingFoot  = curLeft

            self.polynomeX.setParameters(timeInterval,F_k_x,flyingFoot.x,flyingFoot.dx,flyingFoot.ddx)
            self.polynomeY.setParameters(timeInterval,F_k_y,flyingFoot.y,flyingFoot.dy,flyingFoot.ddy)
            self.polynomeQ.setParameters(timeInterval,PreviewAngle,flyingFoot.q,flyingFoot.dq,flyingFoot.ddq)
            for i in range(self.intervaleSize):
                if currentSupport.foot == "left" :
                    flyingFootBuffer = RightFootBuffer
                    supportFootBuffer = LeftFootBuffer
                else :
                    flyingFootBuffer = LeftFootBuffer
                    supportFootBuffer = RightFootBuffer

                # the non swing foot stay still
                supportFootBuffer[i] = deepcopy(supportFoot)
                supportFootBuffer[i].supportFoot = 1

                Ti = self.Tc * i # interpolation time
                Tlocal = localInterpolationStartTime + Ti

                # if we are landing or lifting the foot, do not modify the x,y and theta
                if localInterpolationStartTime < endOfLiftoff:
                    Tr = Ti - endOfLiftoff # Tr = remaining time
                    flyingFootBuffer[i] = self.computeXYQ(flyingFootBuffer[i],Tr)
                else:
                    flyingFootBuffer[i] = self.computeXYQ(flyingFootBuffer[i],Ti)

                flyingFootBuffer[i].z = self.polynomeZ.compute(Tlocal)
                flyingFootBuffer[i].dz = self.polynomeZ.computeDerivative(Tlocal)
                flyingFootBuffer[i].ddz = self.polynomeZ.computeSecDerivative(Tlocal)

            if localInterpolationStartTime < endOfLiftoff:
                # Compute the next iteration state
                flyingFoot = self.computeXYQ(flyingFoot,self.Tc*self.intervaleSize - endOfLiftoff)
            else:
                flyingFoot = self.computeXYQ(flyingFoot,self.Tc*self.intervaleSize)

            return curLeft,curRight,LeftFootBuffer, RightFootBuffer


    def computeXYQ(self,foot,t):
        # compute the foot states at time t
        foot.  x = self.polynomeX.compute(t)
        foot. dx = self.polynomeX.computeDerivative(t)
        foot.ddx = self.polynomeX.computeSecDerivative(t)

        foot.  y = self.polynomeY.compute(t)
        foot. dy = self.polynomeY.computeDerivative(t)
        foot.ddy = self.polynomeY.computeSecDerivative(t)

        foot.  q = self.polynomeQ.compute(t)
        foot. dq = self.polynomeQ.computeDerivative(t)
        foot.ddq = self.polynomeQ.computeSecDerivative(t)

        return foot

class Polynome(object):
    """
    Polynome class of walking pattern generator for humanoids, cf.
    LAAS-UHEI walking report.

    Polynome class provides basic mathematics tools for the interpolation
    """
    def __init__(self,degree):
        self.degree = degree
        self.coef = numpy.zeros( (degree+1,) , dtype=float )

    def compute(self,time):
        if time > self.FT :
            time = self.FT
        if time < 0 :
            time = 0
        r = 0.0; t = 1.0
        for i in range(self.coef.shape[0]):
            r = r + self.coef[i] * t
            t = t * time ;
        return r

    def computeDerivative(self,time):
        if time > self.FT :
            time = self.FT
        if time < 0 :
            time = 0
        r = 0.0; t = 1.0
        for i in range(self.coef.shape[0])[1:]:
            r = r + i* self.coef[i] * t
            t = t * time ;
        return r

    def computeSecDerivative(self,time):
        if time > self.FT :
            time = self.FT
        if time < 0 :
            time = 0
        r = 0.0; t = 1.0
        for i in range(self.coef.shape[0])[2:]:
            r = r + i * (i-1) * self.coef[i] * t
            t = t * time ;
        return r

class Polynome5(Polynome):
    """
    Polynome5 class of walking pattern generator for humanoids, cf.
    LAAS-UHEI walking report.

    Polynome5 class provides all methods to interpolate from a point A
    to a point A with certain velocity an acceleration with a 5th degree
    polynome
    """
    def __init__(self):
        self.FT = 0.0
        self.FP = 0.0
        self.IP = 0.0
        self.IS = 0.0
        self.IA = 0.0
        Polynome.__init__(self,5)
        self.setParameters(self.FT,self.FP,self.IP,self.IS,self.IA)

    def setParameters(self,FinalTime,FinalPosition,InitialPosition,InitialSpeed,InitialAcceleration):
        self.FT = deepcopy(FinalTime)
        self.FP = deepcopy(FinalPosition)
        self.IP = deepcopy(InitialPosition)
        self.IS = deepcopy(InitialSpeed)
        self.IA = deepcopy(InitialAcceleration)

        FT = self.FT
        FP = self.FP
        IP = self.IP
        IS = self.IS
        IA = self.IA

        self.coef = numpy.zeros( (6,) , dtype=float )

        self.coef[0] = IP
        self.coef[1] = IS
        self.coef[2] = IA*0.5

        T = FT*FT*FT
        if T == 0 :
            self.coef[3] = self.coef[4] = self.coef[5] = 0.0
        else :
            self.coef[3] = (-1.5*IA*FT*FT - 6.0*IS*FT - 10.0*IP + 10.0*FP)/T
            T = T * FT
            self.coef[4] = ( 1.5*IA*FT*FT + 8.0*IS*FT + 15.0*IP - 15.0*FP)/T
            T = T * FT
            self.coef[5] = (-0.5*IA*FT*FT - 3.0*IS*FT -  6.0*IP +  6.0*FP)/T

class Polynome4(Polynome):
    """
    Polynome4 class of walking pattern generator for humanoids, cf.
    LAAS-UHEI walking report.

    Polynome4 class provides all methods to interpolate from a point A
    to a point A with certain velocity an acceleration with a 4th degree
    polynome
    """
    def __init__(self):
        self.FT = 0.0
        self.MP = 0.0
        Polynome.__init__(self,4)
        self.setParameters(self.FT,self.MP,0.0,0.0)

    def setParameters(self,FinalTime,MiddlePosition,InitialPosition,InitialSpeed):
        self.FT = deepcopy(FinalTime)
        self.MP = deepcopy(MiddlePosition)
        FT = self.FT
        MP = self.MP
        IP = InitialPosition
        IS = InitialSpeed
        self.coef[0] = deepcopy(IP)
        self.coef[1] = deepcopy(IS)

        T = FT*FT
        if T == 0 :
            self.coef[2] = self.coef[3] = self.coef[4] = 0.0
        else :
            self.coef[2] = (-4.0*IS*FT - 11.0*IP + 16.0*MP)/T
            T = T * FT
            self.coef[3] = ( 5.0*IS*FT + 18.0*IP - 32.0*MP)/T
            T = T * FT
            self.coef[4] = (-2.0*IS*FT -  8.0*IP + 16.0*MP)/T
