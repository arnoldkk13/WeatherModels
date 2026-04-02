
import numpy as np
from abc import ABC, abstractmethod
"""
ChaoticSystem defines the base class for the chaotic systems explored in this project.
This holds the inital state logic (starting x, y, and z), defines the trajectory and error,
and sets the initial conditions. The update steps 
"""
class ChaoticSystem(ABC):
	def __init__(self, initial_state=None):
		if initial_state is None:
			initial_state = np.array([1.0, 1.0, 1.0])
		self.state = initial_state
  
	# Also define the trajectory to hold the states
		self.trajectory = []
		# Define the error to hold the error across timesteps 
		self.error = []
	
	# If defining custom starting coditions, run on setup as well as the init. 
	# Separate for simplicity when setting up in driver, as either x,y,z can be
	# passed into initial_state or as separate params here
	def initial_conditions(self, x, y, z):
		print(f"Setting Initial Conditions: x={x}, y={y}, z={z}")
		self.state =np.array([x, y, z], dtype=float) # Set the state
		self.trajectory = [self.state.copy()]
	
	def get_state(self):
		return self.state.copy()

	"""
	Sets the current state of the system manually.
	This is primarily used in lyapunov computations for 
	normalization
 	"""
	def set_state(self, state):
		self.state = np.array(state, dtype=float)
  

	"""
	Appends the current state to the trajectory.
	"""
	def record_state(self):
		self.trajectory.append(self.state.copy())
 	
	"""
	Compute derivatives for the systems. This is 
	where the main logic for each class differs and
	therefore is abstract here
	"""
	@abstractmethod
	def compute_derivatives(self, state, t=None):
		pass

	@abstractmethod
	def jacobian(self, state):
		pass