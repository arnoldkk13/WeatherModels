from NumericalMethods import RungeKutta
from NumericalMethods import ForwardEuler

import LorenzSystem

class Simulator:
	def __init__(self, system, solver):
		self.system = system
		self.solver = solver

	def run_simulation(self, timesteps):
		for _ in range(timesteps):
			dt, accepted = integrator.step(system, dt)
			if not accepted:
				continue


system = LorenzSystem()
system.initial_coditions(x=0, y=1, z=0) # Setup initial conditions 

integrator = RungeKutta("DormandPrince45")

timesteps = 10000

initial_state = [0,1,0]

for _ in range(timesteps):
	dt, accepted = integrator.step(system, dt)
	if not accepted:
		continue