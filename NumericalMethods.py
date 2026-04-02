"""
NumericalMethods houses the code for numerical methods used in the chaotic
attractor models. This uses the strategy design pattern to choose numerical methods
either at runtime or using a passed in flag.
"""
import numpy as np

"""
Forward Euler (or Explicit Euler) is the simplest method that computes updates based 
off of the current location and the current time step derivative. This is of the form
yn+1 = yn + dt * f(tn, yn), where each new value yn+1 is determined by the previous
plus a step (dt) in the direction of the current derivative f(tn, yn)
"""
class ForwardEuler:
	"""
	Performs one step of the forward euler and updates all states
	"""
	def step(self, system, dt, tol=None):
		dx, dy, dz = system.compute_derivatives(system.state)
		derivative = np.array([dx, dy, dz])
		system.state = system.state + dt * derivative
		system.trajectory.append(system.state.copy())
		return dt, True # The time derivative does not change and the state is always accepted
	
"""
The Dormand-Prince 45 method is a 5th and 4th order Runge-Kutta embedded method that 
uses six functions to calculate fourth and fifth order solutions. These are found from
the butcher tableau defined below. This is the default method when running the simulation.
"""	
class DormandPrince45:
	def __init__(self):
		"""Butcher Tableau for dopri45"""
		self.A = np.array([
		[0, 0, 0, 0, 0, 0, 0],
		[1/5, 0, 0, 0, 0, 0, 0],
		[3/40, 9/40, 0, 0, 0, 0, 0],
		[44/45, -56/15, 32/9, 0, 0, 0, 0],
		[19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
		[9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
		[35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
		])

		self.C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])

		"""4th and 5th Order Coefficients"""
		self.b_high = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
		self.b_low = np.array([5179/57600,0,7571/16695,393/640,-92097/339200,187/2100,1/40])
	
		self.stages = 7
		self.order = 5 

"""
The HeunEuler method is a 2nd and 1st order Runge-Kutta embedded method that can be used
to find an ereror estimate for the forward euler (the 1st order method). These are found from
the butcher tableau defined below.
"""	
class HeunEuler:
	def __init__(self):
		"""Butcher Tableau for Heun-Euler 2(1) method"""
		self.A = np.array([
			[0,   0],
			[1,   0],
			[1/2, 1/2]  # This row is just for Heun's final combination
		])

		self.C = np.array([0, 1, 1])  # Nodes (c)

		"""1st and 2nd Order Coefficients"""
		self.b_high = np.array([1/2, 1/2, 0])  # 2nd order (Heun)
		self.b_low = np.array([1,   0,   0])  # 1st order (Euler)

		self.stages = 3

"""
RungeKutta implements a family of embedded Runge Kutta methods for
solving systems of ODEs. These methods compute derivatives at multiple
intermediate stages within each timestep and combine as designated from their
respective Butcher tableau. 

Embedded methods compute solutions with a higher and lower order estimate, so 
an error estimate can be calculated in the same step. This estimate is used to adjust 
our adaptive timestep, which is defined in this class.

This uses the strategy design pattern to select methods.
"""
class RungeKutta:
	def __init__(self, method, adaptive=True):
		
		self.method = method()
		self.adaptive = adaptive
	"""
	Uses an adaptive step distance based on the error and tolerance we achieve from
	the embedded method. We define some sanity checks such as a minimum timestep of 1e-8 
	and a max of .1.
	"""
	def adaptive_step(self, dt, error, order, tol=1e-3):
		safety = .9
		min_dt = 1e-8  # don't allow dt below this
		max_dt = 1e-1
		# If error is too small to be tracked, increase the timestep 
		if error == 0:
			scale = 2.0
		else:
			scale = safety * (tol / error) ** (1 / (order + 1))
		
		scale = np.clip(scale, 0.1, 5.0)
  
		dt_new = dt * scale
		if abs(dt_new - dt) < 1e-12:
			dt_new = dt * 0.5
		
		# Enforce bounds
		if dt_new < min_dt:
			dt_new = min_dt
		elif dt_new > max_dt:
			dt_new = max_dt
		
		return dt_new
	
	"""
	Performs one step with the specified Runge Kutta method. We return the new
 	modified timestep from adaptive_step and a boolean indicating whether the timestep
	is accepted (scaled error < a tolerance threshold). This is designed to run in a loop
	in the run_simulation function.
	"""
	def step(self, system, dt, tol=1e-3):
		A = self.method.A
		b_high = self.method.b_high
		b_low = self.method.b_low
		stages = self.method.stages
		order = self.method.order

		k = [None] * stages

		for i in range(stages):
			y_temp = system.state.copy()

			for j in range(i):
				y_temp += dt * A[i][j] * k[j]

			k[i] = system.compute_derivatives(y_temp)

		y_high = system.state.copy()
		y_low  = system.state.copy()

		for i in range(stages):
			y_high += dt * b_high[i] * k[i]
			y_low  += dt * b_low[i]  * k[i]

		atol = 1e-6
		rtol = tol 

		scale = atol + rtol * np.maximum(np.abs(y_high), np.abs(system.state))

		error = np.sqrt(np.mean(((y_high - y_low) / scale) ** 2))
  
		### Fixed timestep ###
		if not self.adaptive:
			system.state = y_high
			return dt, True # dt is unchanged, and we always accept the step

		### Adaptive timestep ###
		else:
			dt_new = self.adaptive_step(dt, error, order, tol)
			
			dt_min = 1e-8
			if dt <= dt_min and error >= 1.0:
				# Force accept to break infinite loop
				system.state = y_high
				return dt_new, True

			# Update based on if we use an adaptive step or not
			if error < 1.0:
				system.state = y_high
				return dt_new, True
			else:
				return dt_new, False

	"""
	
	"""
	def step(self, f, state, dt, tol=1e-3):
		A = self.method.A
		b_high = self.method.b_high
		b_low = self.method.b_low
		stages = self.method.stages
		order = self.method.order

		k = [None] * stages

		for i in range(stages):
			y_temp = state.copy()

			for j in range(i):
				y_temp += dt * A[i][j] * k[j]

			k[i] = k[i] = f(y_temp)

		y_high = state.copy()
		y_low  = state.copy()

		for i in range(stages):
			y_high += dt * b_high[i] * k[i]
			y_low  += dt * b_low[i]  * k[i]

		atol = 1e-6
		rtol = tol 

		scale = atol + rtol * np.maximum(np.abs(y_high), np.abs(state))

  
		### Fixed timestep always runs here ###
		
		state = y_high
		return state 