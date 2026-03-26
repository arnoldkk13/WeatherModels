from NumericalMethods import RungeKutta
from NumericalMethods import HeunEuler
from NumericalMethods import DormandPrince45

from NumericalMethods import ForwardEuler

from LorenzSystem import LorenzSystem

from Visualizations import Visualize
from Visualizations import Animate

class Simulator:
	def __init__(self, system, solver, timesteps=None, seconds=None):
		self.system = system
		self.solver = solver
		self.total_time = 0.0
		self.steps = 0

		self.timesteps = timesteps if timesteps else None
		self.seconds = seconds if seconds else None

	"""

	"""
	def run_simulation(self, dt=1e-4, sample_dt=0.01):
		accepted_steps = 0
		times = []
  
		self.total_time = 0.0
  
		sample_dt = 0.01
		next_sample_time = 0.0
		visualization_trajectory = []
  
		if self.seconds: 
			while (self.total_time < self.seconds):
				dt, accepted = solver.step(system, dt)
	
				if not accepted: # If the step has too large error, reduce the timestep size and try again
					continue 
 
				self.total_time += dt
				accepted_steps += 1

				# Downsampling for visualization
				while self.total_time >= next_sample_time:
					visualization_trajectory.append(system.state.copy())
					next_sample_time += sample_dt
	 
				times.append(self.total_time)
				

		else:
			while accepted_steps < self.timesteps:
				dt, accepted = solver.step(system, dt)
	
				if not accepted:
					continue
 
				self.total_time += dt
				accepted_steps += 1

				while self.total_time >= next_sample_time:
					visualization_trajectory.append(system.state.copy())
					next_sample_time += sample_dt
	 
				times.append(self.total_time)
	
		return accepted_steps, times, visualization_trajectory


### MAIN ###
if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--method", type=str, default="DormandPrince45")
	parser.add_argument("--dt", type=float)
	parser.add_argument("--sample_dt", type=float, default=.01) # For visualization purposes we can tweak the update frequency on the graph
	parser.add_argument("--timesteps", type=int, default=10000)
	parser.add_argument("--seconds", type=float) # We can either use seconds or total timesteps
	parser.add_argument("--x", type=float, default=0.0)
	parser.add_argument("--y", type=float, default=1.0)	
	parser.add_argument("--z", type=float, default=0.0)
	parser.add_argument("--visualize", action="store_true", required=False)
	parser.add_argument("--animate", action="store_true", required=False)
	parser.add_argument("--interval", type=int, default=5) # Stores the animation interval time

	args = parser.parse_args()
 
	# Define the system used
	system = LorenzSystem(initial_state=[args.x, args.y, args.z])
 
 	# Define the solver used
  
	dt = args.dt if args.dt else None
	print(f"Using solver {args.method}.\n")

	METHODS = {
		"DormandPrince45": DormandPrince45,
		"HeunEuler": HeunEuler
	}
	# Forward Euler uses a separate class, whereas RungeKutta classes are inherited
	if args.method == "ForwardEuler":
		solver = ForwardEuler()
		print(f"Using steplength {dt}")
	else:
		method_class = METHODS[args.method]
		solver = RungeKutta(method_class)
		print(f"Using variable steplength")
  
	# Setup simulation object
	simulation = Simulator(system=system, solver=solver, timesteps=args.timesteps, seconds=args.seconds)

	# Run simulation. If we are using seconds, we run until total seconds exceeds our seconds.
	# If we are using timesteps, we run until the number of desired timesteps is reached
	if args.timesteps:
		print(f"Running simulation with {args.timesteps} timesteps.\n")
	else:
		print(f"Running simulation for {args.seconds} seconds.\n")
	
	if dt is not None:
		steps, time, trajectory = simulation.run_simulation(dt=args.dt, sample_dt=args.sample_dt)
	else: 
		steps, time, trajectory = simulation.run_simulation(sample_dt=args.sample_dt)
	
	# Visualize the simulation if the flag is used
	print(f"Visualizing the simulation.\n")
	if args.visualize:
		visualizer = Visualize()

		visualizer.plot(steps, time, trajectory, args.method, dt)
  
	# Animate the simulation if the flag is used
	print(f"Animating the simulation.\n")
	if args.animate:
		animator = Animate()
		animator.animate(steps, time, trajectory, args.method, args.sample_dt, args.interval)
