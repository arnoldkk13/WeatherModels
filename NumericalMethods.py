"""
numerical_methods houses the code for numerical methods used in the Lorenz 
Attractor model. This uses the strategy design pattern to choose numerical methods
either at runtime or using a passed in flag.
This is written by Kyle Arnold for Math435
"""
import numpy as np

"""
Forward Euler (or Explicit Euler) is the simplest method that computes updates based 
off of the current location and the current time step derivative.
"""
class ForwardEuler:
	"""
	Performs one step of the forward euler and updates all states
	"""
	def step(self, system, dt):
		derivative = system.compute_derivatives(system.state)
		system.state = system.state + dt * derivative
		system.trajectory.append(system.state.copy())
		return dt, True # The time derivative does not change and the state is always accepted
	
"""
The Dormand-Prince 45 method is a 5th and 4th order Runge-Kutta method
that is used to model the Lorenz system. The
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

class RungeKutta:
	def __init__(self, method):
		self.method = method
	"""
	Uses an adaptive step distance based on the error and tolerance we set up
	"""
	def adaptive_step(self, dt, error, order, tol=1e-6):
		safety = .9
		# If error is too small to be tracked, increase the timestep 
		if error == 0:
			return dt * 2
		
		scale = safety * (tol / error) ** (1/order)
		
		return dt * np.clip(scale, min=.1, max=5.0) 
	
	## TODO: Write the runge_kutta_stepper method that takes in the method name and returns the step
	# That we use for this
	"""
	Performs one step with the specified Runge Kutta method
	"""
	def step(self, system, dt, t, tol=1e-6):
		# Retrieve the butcher tableau for the method
		#      Here, b_low correlates to the array for the lower dimension coefficients and 
		#      b_high correlates to the array for the higher dimension coefficients
		A = self.method.A
		C = self.method.C # Technically there is no time dependence for the Lorenz System but we can include this anyway
		b_high = self.method.b_high 
		b_low = self.method.b_low 
		stages = self.method.stages 
		order = self.method.order
  
		# Retrieve the current locations of the system in normal form
		y = system.state 
		f = system.compute_derivatives
  
		# Store the intermediate derivatives
		k = [np.zeros_like(y) for _ in range(stages)]
  
		for i in range(stages):
			increment = 0
			for j in range(i):
				increment += A[i][j] * k[j]
			t_i = t + C[i] * dt # This step does not run for our example
			k[i] = f(y + dt * increment, t_i)
   
		# Compute high and low order solutions
		y_high = y + dt * sum(b_high[i] * k[i] for i in range(stages))
		y_low = y + dt * sum(b_low[i] * k[i] for i in range(stages))
  
		# Compute error
		scale = 1e-6 + 1e-3 * np.maximum(np.abs(y_high), np.abs(y))
		error = np.sqrt(np.mean(((y_high - y_low) / scale) ** 2))
  
		# Compute new timestep 
		new_dt = self.adaptive_step(dt, error, order)

		if error <= tol:
			# Accept the state
			system.state = y_high 
			system.trajectory.append(system.state.copy())
			system.error.append(error)
			accepted = True 
		else:
			accepted = False 
   
		return new_dt, accepted
