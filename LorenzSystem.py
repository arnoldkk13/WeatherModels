import numpy as np
"""
LorenzSystem defines the code for the Lorenz System, which is
a set of three ODEs defined by Edward Lorenz for atmospheric 
convection.
The system is defined by the following ODEs:
	dx/dt = σ(y-x)
	dy/dt=x(ρ - z)- y
	dz/dt = xy - βz
Where:
	σ is the Prandtl number 
	ρ is the Rayleigh number 
	β is the number of dimensions of the fluid layer
"""
class LorenzSystem:
	
	"""
	Initializes the Lorenz System coefficients
	"""
	def __init__(self, initial_state=None, σ=10, ρ=28, β=8/3):
		# Define coefficients for σ, ρ, and β by the passed in values
		print(f"Initial Coefficients: σ={σ}, ρ={ρ}, β={β}")
		self.σ = σ 
		self.ρ = ρ
		self.β = β
	
		# Define the initial state
		if initial_state is None:
			initial_state = np.array([1.0, 1.0, 1.0])
		self.state = initial_state.astype(float)

		# Also define the trajectory to hold the states
		self.trajectory = []
		# Define the error to hold the error across timesteps 
		self.error = []
	
	# Must run on setup as well as the init. Separate for simplicity when setting up in driver
	def initial_conditions(self, x, y, z):
		print(f"Initial Conditions: x={x}, y={y}, z={z}")
		self.x = x
		self.y = y 
		self.z = z
		self.state = [self.x, self.y, self.z]
  
	# Fetches the most up to date state
	def update_state(self):
		self.state = [self.x, self.y, self.z]
  
	def compute_derivatives(self, state, t=None):
		x, y, z = state
  
		dx = self.σ * (y - x)
		dy = x * (self.ρ - z) - y
		dz = x * y - self.β * z

		return np.array[dx, dy, dz]