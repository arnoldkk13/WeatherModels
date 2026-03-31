from ChaoticSystem import ChaoticSystem
import numpy as np
"""
AizawaAttractor defines the code for the Aizawa System, which is
a set of three ODEs that produce a complex spherical attractor.
The system is defined by the following ODEs:
	dx/dt = (z - b)x - d*y
    dy/dt = d*x + (z - b)y
    dz/dt = c + a*z - (z^3/3) - (x^2 + y^2)(1 + e*z) + f*z*x^3
"""
class AizawaAttractor(ChaoticSystem):
	"""
	Initializes the Aizawa System coefficients
	"""
	def __init__(self, initial_state=None, a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1):
		super().__init__(initial_state)
		# Define coefficients for initial coefficients by the passed in values
		print(f"Aizawa Coefficients: a={a}, b={b}, c={c}, d={d}, e={e}, f={f}")
		self.a = a
		self.b = b
		self.c = c
		self.d = d
		self.e = e
		self.f = f

	"""
	Computes the Aizawa system derivatives dx, dy, and dz
	given the state x, y, z.
	This is found from the system definition:
	dx/dt = (z - b)x - d*y
    dy/dt = d*x + (z - b)y
    dz/dt = c + a*z - (z^3/3) - (x^2 + y^2)(1 + e*z) + f*z*x^3
	"""
	def compute_derivatives(self, state):
		x, y, z = state
  
		dx = (z - self.b) * x - self.d * y
		dy = self.d * x + (z - self.b) * y
		dz = self.c + self.a * z - (z**3 / 3) - (x**2 + y**2) * (1 + self.e * z) + self.f * z * x**3

		return np.array([dx, dy, dz])