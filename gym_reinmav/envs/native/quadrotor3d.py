# **********************************************************************
#
# Copyright (c) 2019, Autonomous Systems Lab
# Author: Jaeyoung Lim <jalim@student.ethz.ch>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# *************************************************************************
import gym
from gym import error, spaces, utils, logger
from math import cos, sin, pi, atan2
import math
import numpy as np
from numpy import linalg
from gym.utils import seeding
from pyquaternion import Quaternion

def quaternion_to_euler(x, y, z, w):

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        X = math.degrees(math.atan2(t0, t1))
        X = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = math.degrees(math.asin(t2))
        Y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Z = math.degrees(math.atan2(t3, t4))
        Z = math.atan2(t3, t4)
        return X, Y, Z


class Quadrotor3D(gym.Env):
	metadata = {'render.modes': ['human']}
	def __init__(self):
		self.mass = 1.535
		self.dt = 0.03
		self.g = np.array([0.0, 0.0, -9.81])

		self.state = None

		self.ref_pos = np.array([0.0, 0.0, 20.0])
		self.ref_vel = np.array([0.0, 0.0, 0.0])
		self.ref_att = np.array([0.0, 0.0, 0.0, 1.0])

        # Conditions to fail the episode
		self.pos_threshold = 10.0
		self.vel_threshold = 10.0

		self.viewer = None
		self.render_quad1 = None
		self.render_quad2 = None
		self.render_rotor1 = None
		self.render_rotor2 = None
		self.render_rotor3 = None
		self.render_rotor4 = None
		self.render_velocity = None
		self.render_ref = None
		self.x_range = 1.0
		self.steps_beyond_done = None

		####################################################
		self.min_action = np.array([0, -1, -1, -1])
		self.max_action = np.array([1, 1, 1, 1])

		self.min_lin_pos = np.array([0.1, -20, -20])
		self.max_lin_pos = np.array([40, 20, 20])

		self.min_lin_vel = 10*np.array([-1, -1, -1])
		self.max_lin_vel = 10*np.array([1, 1, 1])

		self.min_lin_accl = 30*np.array([-1, -1, -1])
		self.max_lin_accl = 30*np.array([1, 1, 1])

		self.min_ang_pos = np.array([-1.57, -1.57, -1.57])
		self.max_ang_pos = np.array([1.57, 1.57, 1.57])

		self.min_ang_vel = 10*np.array([-1, -1, -1])
		self.max_ang_vel = 10*np.array([1, 1, 1])

		self.low_state = np.array([ self.min_lin_pos, self.min_lin_vel, self.min_lin_accl, self.min_ang_pos, self.min_ang_vel ]).flatten()
		self.high_state = np.array([ self.max_lin_pos, self.max_lin_vel, self.max_lin_accl, self.max_ang_pos,  self.max_ang_vel ]).flatten()

		self.action_space = spaces.Box(low = self.min_action, high=self.max_action, dtype=np.float32)
		self.observation_space = spaces.Box(low = self.low_state, high = self.high_state, dtype = np.float32)

		###################################################################3

		self.real_action_high = np.array([30.0,2.0,2.0,2.0])

		self.real_action_low = np.array([0.0,-2.0,-2.0,-2.0])

		# action_low = np.array([0.0])
		# action_high = np.array([30.0])
		
		## observation_space = [posx,poy,posz,quat1,quat2,quat3,quat4,velx,vely,velz]

		# observation_low = np.array([-30,-30,0,-1,-1,-1,-1,-10,-10,-10])
		# observation_high = np.array([30,30,30,1,1,1,1,10,10,10])
		# observation_low = np.array([-30,-30,0,-10,-10,-10])
		# observation_high = np.array([30,30,30,10,10,10])

		# observation_low = np.array([0,-30])
		# observation_high = np.array([30,30])

		# self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float )
		# self.observation_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.float)
		
		self.seed()
		self.reset()


	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):
		norm_thrust = action[0] # Thrust command

		thrust = norm_thrust * ( self.real_action_high[0] - self.real_action_low[0] )/( self.max_action[0]-self.min_action[0] )
				
		norm_w = action[1:4] # Angular velocity command

		w = norm_w * (self.real_action_high[1:4]-self.real_action_low[1:4])/(self.max_action[1:4]-self.min_action[1:4])

		state = self.state
		ref_pos = self.ref_pos
		ref_vel = self.ref_vel
		ref_att = self.ref_att

		pos = np.array([state[0], state[1], state[2]]).flatten()
		att = np.array([state[3], state[4], state[5], state[6]]).flatten()
		vel = np.array([state[7], state[8], state[9]]).flatten()


		att_quaternion = Quaternion(att)

		acc = thrust/self.mass * att_quaternion.rotation_matrix.dot(np.array([0.0, 0.0, 1.0])) + self.g
		
		pos = pos + vel * self.dt + 0.5*acc*self.dt*self.dt
		vel = vel + acc * self.dt
		
		q_dot = att_quaternion.derivative(w)
		att = att + q_dot.elements * self.dt

		state = (pos[0], pos[1], pos[2], att[0], att[1], att[2], att[3], vel[0], vel[1], vel[2])
		# state = (pos[0], pos[1], pos[2], vel[0], vel[1], vel[2])

		self.state = np.array(state)

		done = np.abs(pos[0]) > self.pos_threshold or np.abs(pos[1]) > self.pos_threshold \
			or pos[2] < self.ref_pos[2]-self.pos_threshold or pos[2] > self.ref_pos[2]+self.pos_threshold
		done = bool(done)

		if not done:
		    max_reward = np.sum(np.power(self.pos_threshold,2))
		    reward = max_reward - np.sum( np.power( pos-self.ref_pos,2 ) ) + 42 - 16*np.sum( np.power( norm_w,2 ) )
		else:
		    reward = -50000
		
		if self.ep_steps > 1000:
			reward += 100
			done = True


		eul = quaternion_to_euler(state[3], state[4], state[5], state[6])

		# ob = np.array([ np.array(pos)-self.ref_pos , np.array(vel) , np.array(eul) ]).flatten()
		ob = np.array([ np.array(pos)-self.ref_pos , np.array(vel) ,np.array(acc), np.array(eul), np.array(w) ]).flatten()

		self.ep_steps+=1
		return ob, reward, done, {}

	def reset(self):

		self.ep_steps = 0

		ini_pos = [0,0,20]#+np.random.randint(-5,6,3)
		ini_vel = [0,0,0]
		ini_quat = [0,0,0,1]
		ini_acc = [0,0,0]
		ini_w = [0,0,0]

		# self.state = np.array([0,0,3,0,0,0,1,0,0,0])
		self.state = np.hstack([ini_pos,ini_quat,ini_vel])

		ini_eul = quaternion_to_euler(ini_quat[0], ini_quat[1], ini_quat[2], ini_quat[3])

		# ob = np.array([ini_pos-self.ref_pos,ini_vel,ini_eul]).flatten()
		ob = np.array([ini_pos-self.ref_pos,ini_vel,ini_acc,ini_eul,ini_w]).flatten()
		return ob

	def render(self, mode='human', close=False):
		from vpython import box, sphere, color, vector, rate, canvas, cylinder, arrow, curve

		def make_grid(unit, n):
			nunit = unit * n
			for i in range(n+1):
				if i%5==0:
					lcolor = vector(0.5,0.5,0.5)
				else:
					lcolor = vector(0.5, 0.5, 0.5)
				curve(pos=[(0,i*unit,0), (nunit, i*unit, 0)],color=lcolor)
				curve(pos=[(i*unit,0,0), (i*unit, nunit, 0)],color=lcolor)
				curve(pos=[(0,-i*unit,0), (-nunit, -i*unit, 0)],color=lcolor)
				curve(pos=[(-i*unit,0,0), (-i*unit, -nunit, 0)],color=lcolor)
				curve(pos=[(0,i*unit,0), (-nunit, -i*unit, 0)],color=lcolor)
				curve(pos=[(i*unit,0,0), (-i*unit, -nunit, 0)],color=lcolor)				

		state = self.state
		ref_pos = self.ref_pos
		ref_vel = self.ref_vel

		pos = np.array([state[0], state[1], state[2]]).flatten()
		att = np.array([state[3], state[4], state[5], state[6]]).flatten()
		vel = np.array([state[7], state[8], state[9]]).flatten()


		current_quat = Quaternion(att)
		x_axis = current_quat.rotation_matrix.dot(np.array([1.0, 0.0, 0.0]))
		y_axis = current_quat.rotation_matrix.dot(np.array([0.0, 1.0, 0.0]))
		z_axis = current_quat.rotation_matrix.dot(np.array([0.0, 0.0, 1.0]))

		if self.viewer is None:
			self.viewer = canvas(title='Quadrotor 3D', width=1024, height=768, center=vector(self.ref_pos[0], self.ref_pos[1], self.ref_pos[2]), forward=vector(1, 1, -0.5), up=vector(0, 0, 1), background=color.white, range=4.0, autoscale = False)
			self.render_quad1 = box(canvas = self.viewer, pos=vector(pos[0],pos[1],0), axis=vector(x_axis[0],x_axis[1],x_axis[2]), length=0.2, height=0.05, width=0.05)
			self.render_quad2 = box(canvas = self.viewer, pos=vector(pos[0],pos[1],0), axis=vector(y_axis[0],y_axis[1],y_axis[2]), length=0.2, height=0.05, width=0.05)
			self.render_rotor1 = cylinder(canvas = self.viewer, pos=vector(pos[0],pos[1],0), axis=vector(0.01*z_axis[0],0.01*z_axis[1],0.01*z_axis[2]), radius=0.2, color=color.cyan, opacity=0.5)
			self.render_rotor2 = cylinder(canvas = self.viewer, pos=vector(pos[0],pos[1],0), axis=vector(0.01*z_axis[0],0.01*z_axis[1],0.01*z_axis[2]), radius=0.2, color=color.cyan, opacity=0.5)
			self.render_rotor3 = cylinder(canvas = self.viewer, pos=vector(pos[0],pos[1],0), axis=vector(0.01*z_axis[0],0.01*z_axis[1],0.01*z_axis[2]), radius=0.2, color=color.cyan, opacity=0.5)
			self.render_rotor4 = cylinder(canvas = self.viewer, pos=vector(pos[0],pos[1],0), axis=vector(0.01*z_axis[0],0.01*z_axis[1],0.01*z_axis[2]), radius=0.2, color=color.cyan, opacity=0.5)
			self.render_velocity = pointer = arrow(pos=vector(pos[0],pos[1],0), axis=vector(vel[0],vel[1],vel[2]), shaftwidth=0.05, color=color.green)
			self.render_ref = sphere(canvas = self.viewer, pos=vector(ref_pos[0], ref_pos[1], ref_pos[2]), radius=0.1, color=color.blue, make_trail = True)
			grid_xy = make_grid(5, 100)
		if self.state is None: return None

		self.render_quad1.pos.x = pos[0]
		self.render_quad1.pos.y = pos[1]
		self.render_quad1.pos.z = pos[2]
		self.render_quad2.pos.x = pos[0]
		self.render_quad2.pos.y = pos[1]
		self.render_quad2.pos.z = pos[2]
		rotor_pos = 0.5*x_axis
		self.render_rotor1.pos.x = pos[0] + rotor_pos[0]
		self.render_rotor1.pos.y = pos[1] + rotor_pos[1]
		self.render_rotor1.pos.z = pos[2] + rotor_pos[2]
		rotor_pos = (-0.5)*x_axis
		self.render_rotor2.pos.x = pos[0] + rotor_pos[0]
		self.render_rotor2.pos.y = pos[1] + rotor_pos[1]
		self.render_rotor2.pos.z = pos[2] + rotor_pos[2]
		rotor_pos = 0.5*y_axis
		self.render_rotor3.pos.x = pos[0] + rotor_pos[0]
		self.render_rotor3.pos.y = pos[1] + rotor_pos[1]
		self.render_rotor3.pos.z = pos[2] + rotor_pos[2]
		rotor_pos = (-0.5)*y_axis
		self.render_rotor4.pos.x = pos[0] + rotor_pos[0]
		self.render_rotor4.pos.y = pos[1] + rotor_pos[1]
		self.render_rotor4.pos.z = pos[2] + rotor_pos[2]
		self.render_velocity.pos.x = pos[0]
		self.render_velocity.pos.y = pos[1]
		self.render_velocity.pos.z = pos[2]

		self.render_quad1.axis.x = x_axis[0]
		self.render_quad1.axis.y = x_axis[1]	
		self.render_quad1.axis.z = x_axis[2]
		self.render_quad2.axis.x = y_axis[0]
		self.render_quad2.axis.y = y_axis[1]
		self.render_quad2.axis.z = y_axis[2]
		self.render_rotor1.axis.x = 0.01*z_axis[0]
		self.render_rotor1.axis.y = 0.01*z_axis[1]
		self.render_rotor1.axis.z = 0.01*z_axis[2]
		self.render_rotor2.axis.x = 0.01*z_axis[0]
		self.render_rotor2.axis.y = 0.01*z_axis[1]
		self.render_rotor2.axis.z = 0.01*z_axis[2]
		self.render_rotor3.axis.x = 0.01*z_axis[0]
		self.render_rotor3.axis.y = 0.01*z_axis[1]
		self.render_rotor3.axis.z = 0.01*z_axis[2]
		self.render_rotor4.axis.x = 0.01*z_axis[0]
		self.render_rotor4.axis.y = 0.01*z_axis[1]
		self.render_rotor4.axis.z = 0.01*z_axis[2]
		self.render_velocity.axis.x = 0.5 * vel[0]
		self.render_velocity.axis.y = 0.5 * vel[1]
		self.render_velocity.axis.z = 0.5 * vel[2]


		self.render_quad1.up.x = z_axis[0]
		self.render_quad1.up.y = z_axis[1]
		self.render_quad1.up.z = z_axis[2]
		self.render_quad2.up.x = z_axis[0]
		self.render_quad2.up.y = z_axis[1]
		self.render_quad2.up.z = z_axis[2]
		self.render_rotor1.up.x = y_axis[0]
		self.render_rotor1.up.y = y_axis[1]
		self.render_rotor1.up.z = y_axis[2]
		self.render_rotor2.up.x = y_axis[0]
		self.render_rotor2.up.y = y_axis[1]
		self.render_rotor2.up.z = y_axis[2]
		self.render_rotor3.up.x = y_axis[0]
		self.render_rotor3.up.y = y_axis[1]
		self.render_rotor3.up.z = y_axis[2]
		self.render_rotor4.up.x = y_axis[0]
		self.render_rotor4.up.y = y_axis[1]
		self.render_rotor4.up.z = y_axis[2]



		self.render_ref.pos.x = ref_pos[0]
		self.render_ref.pos.y = ref_pos[1]
		self.render_ref.pos.z = ref_pos[2]

		rate(100)

		return True

	def close(self):
		if self.viewer:
			self.viewer = None
			
	def control(self):
			def acc2quat(desired_acc, yaw): # TODO: Yaw rotation
				zb_des = desired_acc / linalg.norm(desired_acc)
				yc = np.array([0.0, 1.0, 0.0])
				xb_des = np.cross(yc, zb_des)
				xb_des = xb_des / linalg.norm(xb_des)
				yb_des = np.cross(zb_des, xb_des)
				zb_des = zb_des / linalg.norm(zb_des)

				rotmat = np.array([[xb_des[0], yb_des[0], zb_des[0]],
								[xb_des[1], yb_des[1], zb_des[1]],
								[xb_des[2], yb_des[2], zb_des[2]]])

				desired_att = Quaternion(matrix=rotmat)

				return desired_att

			Kp = np.array([-5.0, -5.0, -5.0])
			Kv = np.array([-4.0, -4.0, -4.0])
			tau = 0.3

			state = self.state
			ref_pos = self.ref_pos
			ref_vel = self.ref_vel

			pos = np.array([state[0], state[1], state[2]]).flatten()
			att = np.array([state[3], state[4], state[5], state[6]]).flatten()
			vel = np.array([state[7], state[8], state[9]]).flatten()

			error_pos = pos - ref_pos
			error_vel = vel - ref_vel

			# %% Calculate desired acceleration
			reference_acc = np.array([0.0, 0.0, 0.0])
			feedback_acc = Kp * error_pos + Kv * error_vel 

			desired_acc = reference_acc + feedback_acc - self.g

			desired_att = acc2quat(desired_acc, 0.0)

			desired_quat = Quaternion(desired_att)
			current_quat = Quaternion(att)

			error_att = current_quat.conjugate * desired_quat
			qe = error_att.elements

			
			w = (2/tau) * np.sign(qe[0])*qe[1:4]

			
			thrust = desired_acc.dot(current_quat.rotation_matrix.dot(np.array([0.0, 0.0, 1.0])))
			
			action = np.array([thrust, w[0], w[1], w[2]])

			return action