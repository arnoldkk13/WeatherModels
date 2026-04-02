from ChaoticSystem import ChaoticSystem
import numpy as np
"""
LorenzAttractor defines the code for the Lorenz System, which is
a set of three ODEs defined by Edward Lorenz for atmospheric 
convection.
The system is defined by the following ODEs:
	dx/dt = σ(y-x)
	dy/dt = x(ρ - z)- y
	dz/dt = xy - βz
Where:
	σ is the Prandtl number 
	ρ is the Rayleigh number 
	β is the number of dimensions of the fluid layer
"""
class LorenzAttractor(ChaoticSystem):
	"""
	Initializes the Lorenz System coefficients
	"""
	def __init__(self, initial_state=None, σ=10, ρ=28, β=8/3):
		super().__init__(initial_state)
		# Define coefficients for σ, ρ, and β by the passed in values
		print(f"Lorenz Coefficients: σ={σ}, ρ={ρ}, β={β}")
		self.σ = σ 
		self.ρ = ρ
		self.β = β

	"""
	Computes the Lorenz system derivatives dx, dy, and dz
	given the state x, y, z.
	This is found from the system definition:
	dx/dt = σ(y-x)
	dy/dt = x(ρ - z)- y
	dz/dt = xy - βz
	"""
	def compute_derivatives(self, state):
		x, y, z = state
  
		dx = self.σ * (y - x)
		dy = x * (self.ρ - z) - y
		dz = x * y - self.β * z

		return np.array([dx, dy, dz])	

	def jacobian(self, state):
		x, y, z = state
		J = np.array([
		[-self.σ, self.σ, 0],
		[self.ρ - z, -1, -x],
		[y, x, -self.β]
		])
		return J