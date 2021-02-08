import os
import sys
import numpy

from math import cos, sin
from time import gmtime, strftime

class FiniteStateMachine(object):
    """
    Finite state machine to implement starting and stopping maneuvers for the
    pattern generator from CNRS-LAAS C++ implementation.
    Finite state machine to determine the support parameters.

    Copyright 2010,

    Andrei  Herdt
    Olivier Stasse

    JRL, CNRS/AIST

    This file is part of walkGenJrl.
    walkGenJrl is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    walkGenJrl is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Lesser Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with walkGenJrl.  If not, see <http://www.gnu.org/licenses/>.

    Research carried out within the scope of the
    Joint Japanese-French Robotics Laboratory (JRL)
    """
    def __init__(self):
        # Precision constant
        self.EPS_ = 1e-6

        # Rotation phase
        # True if the robot is in translation
        self.InTranslation_ = False

        # True if the robot is in rotation
        self.InRotation_ = False

        # Number of stabilize steps after the end of a rotation
        self.NbStepsAfterRotation_ = 0

        # Current support foot type (SS, DS)
        self.CurrentSupportFoot_ = 'left'

        # True if the end phase of the rotation has begun
        self.PostRotationPhase_ = False

        # Length of a step
        self.StepPeriod_ = 0.8

        # Length of a double support phase
        self.DSPeriod_ = 1e9

        # Duration of the transition ds -> ss
        self.DSSSPeriod_ = 0.8

        # Number of steps to be done before DS
        self.NbStepsSSDS_ = 200

        # Sampling period
        self.T_ = 0.005

    def update_vel_reference(Ref, CurrentSupport):
        """
        Update the velocity reference after a pure rotation

        Parameters
        ----------

        Ref: numpy.ndarray((3,), dtype=float)
            new reference velocity

        CurrentSupport: str
            current support foot state
        """
        # Check if the robot is supposed to translate
        if abs(Ref.Local.X)>2*self.EPS_ or abs(Ref.Local.Y)>2*self.EPS_:
            self.InTranslation_ = True
        else:
            self.InTranslation_ = False

        # Check if the robot is supposed to rotate
        if abs(Ref.Local.Yaw) > self.EPS_:
            self.InRotation_ = True
        # Check if he is still in motion
        else:
            if self.InRotation_ and not self.InTranslation_:
                Ref.Local.X = 2*self.EPS_ # @Max: why does it change velocity?
                Ref.Local.Y = 2*self.EPS_ #       that it's set to InTranslation_ ?
                if not self.PostRotationPhase_:
                    self.CurrentSupportFoot_ = CurrentSupport.Foot
                    self.NbStepsAfterRotation_ = 0
                    self.PostRotationPhase_ = True
                else:
                    if self.CurrentSupportFoot_ != CurrentSupport.Foot:
                        self.CurrentSupportFoot_    = CurrentSupport.Foot
                        self.NbStepsAfterRotation_ += 1

                    if self.NbStepsAfterRotation_>2:
                        self.InRotation_        = False
                        self.PostRotationPhase_ = False
            else:
                self.InRotation_ = False

    def set_support_state(time, pi, Support, Ref):
        """
        Initialize the previewed state

        Parameters
        ----------

        time: float
            current time

        pi: float
            Number of (p)reviewed sampling (i)nstant inside the preview period

        Support: str
            current support foot state

        Ref: numpy.ndarray((3,), dtype=float)
            new reference velocity
        """

        Support.StateChanged = False
        Support.NbInstants  += 1

        ReferenceGiven = False
        if fabs(Ref.Local.X)   > EPS_ \
        or fabs(Ref.Local.Y)   > EPS_ \
        or fabs(Ref.Local.Yaw) > EPS_:
            ReferenceGiven = True

        # Update time limit for double support phase
        if  ReferenceGiven and Support.Phase == DS \
        and (Support.TimeLimit-time-EPS_) > DSSSPeriod_:
            Support.TimeLimit = time+DSSSPeriod_
            Support.NbStepsLeft = NbStepsSSDS_

        # FSM
        if time+EPS_+pi*T_ >= Support.TimeLimit:
            # SS->DS
            if  Support.Phase == SS  and not ReferenceGiven \
            and Support.NbStepsLeft == 0:
                Support.Phase = DS
                Support.TimeLimit = time+pi*T_+DSPeriod_
                Support.StateChanged = True
                Support.NbInstants = 0

        # DS->SS
        elif ((Support.Phase == DS) and ReferenceGiven) \
        or   ((Support.Phase == DS) and (Support.NbStepsLeft > 0)):
            Support.Phase = SS
            Support.TimeLimit = time+pi*T_+StepPeriod_
            Support.NbStepsLeft = NbStepsSSDS_
            Support.StateChanged = True
            Support.NbInstants = 0

        # SS->SS
        elif (Support.Phase == SS and Support.NbStepsLeft > 0) \
        or   (Support.NbStepsLeft == 0 and ReferenceGiven):
            if Support.Foot == 'left':
                Support.Foot = 'right'
            else:
              Support.Foot = 'left'

            Support.StateChanged = True
            Support.NbInstants = 0
            Support.TimeLimit = time+pi*T_+StepPeriod_
            # Flying foot is not down
            if pi != 1:
                Support.StepNumber += 1
            if not ReferenceGiven:
                Support.NbStepsLeft = Support.NbStepsLeft-1
            if ReferenceGiven:
                Support.NbStepsLeft = NbStepsSSDS_


class BaseTypeSupportFoot(object):
    """
    """

    def __init__(self, x=0, y=0, theta=0, foot="left"):
        self.x = x
        self.y = y
        self.q = theta
        self.foot = foot
        self.ds = 0
        self.stepNumber = 0
        self.timeLimit = 0

    def __eq__(self, other):
        """ equality operator to check if A == B """
        return (isinstance(other, self.__class__)
            or self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)


class BaseTypeFoot(object):
    """
    """
    def __init__(self, x=0, y=0, theta=0, foot="left", supportFoot=0):
        self.x = x
        self.y = y
        self.z = 0
        self.q = theta

        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.dq = 0

        self.ddx = 0
        self.ddy = 0
        self.ddz = 0
        self.ddq = 0

        self.supportFoot = supportFoot

    def __eq__(self, other):
        """ equality operator to check if A == B """
        return (isinstance(other, self.__class__)
            or self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)


class CoMState(object):

    def __init__(self, x=0, y=0, theta=0, h_com=0.814):
        self.x = numpy.zeros( (3,) , dtype=float )
        self.y = numpy.zeros( (3,) , dtype=float )
        self.z = h_com
        self.q = numpy.zeros( (3,) , dtype=float )


class ZMPState(object):
    """
    """
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other):
        """ equality operator to check if A == B """
        return (isinstance(other, self.__class__)
            or self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

