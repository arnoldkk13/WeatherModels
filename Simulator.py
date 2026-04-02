from NumericalMethods import RungeKutta
from NumericalMethods import HeunEuler
from NumericalMethods import DormandPrince45

from NumericalMethods import ForwardEuler

from LorenzAttractor import LorenzAttractor
from RosslerAttractor import RosslerAttractor
from ChuaCircuit import ChuaCircuit
from AizawaAttractor import AizawaAttractor

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
	Runs the simulation with a given tolerance. Can either be run with
	a total number of desired seconds or a number of timesteps due to our
	adaptive timestep method. This also downsamples our visualization for
	performance purposes and stores the values in system.trajectory.
	
	"""
	def run_simulation(self, dt=.001, sample_dt=0.01, tol=1e-3):
		accepted_steps = 0
		times = []
  
		self.total_time = 0.0
		next_sample_time = 0.0
		next_percent_to_print = 5 # First print at 5%
		not_accepted_counter = 0
		max_retries = 20
  
		if self.seconds: 
			while (self.total_time < self.seconds):
				dt, accepted = solver.step(system, dt, tol)
	
				if not accepted:
					not_accepted_counter += 1

					if not_accepted_counter >= max_retries:
						print("Forcing step acceptance after repeated failures")
						system.state = system.state  # or keep last y_high if available
						accepted = True
						not_accepted_counter = 0
					else:
						continue
 
				self.total_time += dt
				accepted_steps += 1

				# Downsampling for visualization
				while self.total_time >= next_sample_time:
					system.record_state() # Updates the trajectory in system with the current state
					next_sample_time += sample_dt
     
				# Progress updates
				progress = self.total_time / self.seconds * 100
				if progress >= next_percent_to_print:
					print(f"Simulation progress: {int(progress)}%")
					next_percent_to_print += 5
	 
				times.append(self.total_time)
				
		else:
			## Timestep functionality is not required necessarily, and may be depreciated ##
			while accepted_steps < self.timesteps:
				dt, accepted = solver.step(system, dt)
	
				if not accepted:
					not_accepted_counter += 1

					if not_accepted_counter >= max_retries:
						print("Forcing step acceptance after repeated failures")
						system.state = system.state  # or keep last y_high if available
						accepted = True
						not_accepted_counter = 0
					else:
						continue
 
				self.total_time += dt
				accepted_steps += 1

				# Downsampling for visualization
				while self.total_time >= next_sample_time:
					system.record_state() # Updates the trajectory in system with the current state
					next_sample_time += sample_dt
     
				# Progress updates
				progress = self.total_time / self.seconds * 100
				if progress >= next_percent_to_print:
					print(f"Simulation progress: {int(progress)}%")
					next_percent_to_print += 5
	 
				times.append(self.total_time)
	
		return accepted_steps, times, system.trajectory


### MAIN ###
if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--method", type=str, default="DormandPrince45")
	parser.add_argument("--system", type=str, default="LorenzAttractor")
	parser.add_argument("--dt", type=float)
	parser.add_argument("--sample_dt", type=float, default=.01) # For visualization purposes we can tweak the update frequency on the graph
	parser.add_argument("--tol", type=float, default=1e-4) # Tweak the tolerance for different chaotic systems
	parser.add_argument("--fixed", action='store_true', required=False) # Fixes the timestep for RK methods, with no adaptive logic
	parser.add_argument("--timesteps", type=int, default=10000)
	parser.add_argument("--seconds", type=float) # We can either use seconds or total timesteps
	parser.add_argument("--x", type=float, default=0.0)
	parser.add_argument("--y", type=float, default=1.0)	
	parser.add_argument("--z", type=float, default=0.0)
	parser.add_argument("--visualize", action="store_true", required=False)
	parser.add_argument("--animate", action="store_true", required=False)
	parser.add_argument("--update_method", type=str, default="linear")
	parser.add_argument("--save", action='store_true', required=False)

	args = parser.parse_args()

	if args.system == "LorenzAttractor":
		# Define the system used
		system = LorenzAttractor(initial_state=[args.x, args.y, args.z])
	elif args.system == "RosslerAttractor":
		system = RosslerAttractor(initial_state=[args.x, args.y, args.z])
	elif args.system == "ChuaCircuit":
		system = ChuaCircuit(initial_state=[args.x, args.y, args.z])
	elif args.system == "AizawaAttractor":
		system = AizawaAttractor(initial_state=[args.x, args.y, args.z])
	else:
		print("Incorrect system used. Exitting")
		exit(1)
 
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
		adaptive = not args.fixed
		method_class = METHODS[args.method]
		solver = RungeKutta(method_class, adaptive)
		print(f"Using variable steplength")
  
	# Setup simulation object
	simulation = Simulator(system=system, solver=solver, timesteps=args.timesteps, seconds=args.seconds)

	# Run simulation. If we are using seconds, we run until total seconds exceeds our seconds.
	# If we are using timesteps, we run until the number of desired timesteps is reached
	if args.seconds:
		print(f"Running simulation for {args.seconds} seconds.\n")
	else:
		print(f" {args.timesteps} timesteps.\n")
	
	if dt is not None:
		steps, time, trajectory = simulation.run_simulation(dt=args.dt, sample_dt=args.sample_dt, tol=args.tol)
	else: 
		steps, time, trajectory = simulation.run_simulation(sample_dt=args.sample_dt, tol=args.tol)
	
	# Visualize the simulation if the flag is used
	if args.visualize:
		print(f"Visualizing the simulation.\n")
		visualizer = Visualize()

		visualizer.plot(steps, time, trajectory, args.method, args.system, dt)
  
	# Animate the simulation if the flag is used
	if args.animate:
		print(f"Animating the simulation.\n")
		animator = Animate()
		animator.animate(steps=steps, time=time, trajectory=trajectory, method_name=args.method, sample_dt=args.sample_dt, system=args.system, update_method=args.update_method, save=args.save)
