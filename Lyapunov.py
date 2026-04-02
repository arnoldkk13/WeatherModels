"""
This code calculates the Largest Lyapunov exponent λ for chaotic systems. Here,
λ > 0 in practice for chaotic systems, so our solution should reflect this behavior.

The Lyapunov exponent quantifies the exponential rate of separation of infinitesimally 
close trajectories. This is quantitatively measured using two trajectories with an initial
separate vector δ0, with the separation δ(t) expressed as follows: |δ(t| ~= e^(λt)|δ0|

This code runs simulations with slightly different initial starting points (x0 and (x0 + δ0)) and calculates the
Lyapunov exponent.

This code was written by Kyle Arnold for Math435 at CSU
"""
from NumericalMethods import RungeKutta
from NumericalMethods import HeunEuler
from NumericalMethods import DormandPrince45

from NumericalMethods import ForwardEuler

from LorenzAttractor import LorenzAttractor
from RosslerAttractor import RosslerAttractor
from ChuaCircuit import ChuaCircuit
from AizawaAttractor import AizawaAttractor



import numpy as np

"""
Lyapunov houses the calculations for the Lyapunov exponent(s). 

The wolf algorithm (calculate_largest_lyapunov) only calculates the Lyapunov 
exponent λ that has the greatest influence on the system, which is the largest.

The bennetin algorithm (calculate_lyapunov) calculates all Lyapunov exponents for the dimension 
of the chaotic system (in our case, always three). This algorithm takes approximately
3x as long to run as the wolf algorithm.
"""
class Lyapunov():
	"""
	Calculates the largest Lyapunov exponent λ for chaotic systems.
	Takes in initialized systems 1 and 2 for adding the small difference ε. This 
	will then run the simulations and calculate the difference in trajectories
	using each system.trajectory values.
 
	"""
	def calculate_largest_lyapunov(seconds, system1, system2, solver, ε=1e-8, dt=1e-3):
		next_percent_to_print = 5 # First print at 5%

		total_time = 0
		n_steps = round(seconds / dt)
		log_sum = 0.0
		# For each timestep in the tital number of timesteps:
		for i in range(n_steps):
			# Step with all systems
			# Base system
			solver.step(system1, dt)
			# Perturbed system
			solver.step(system2, dt)

			# Compute the current distance between the vectors 
			δ = system2.get_state() - system1.get_state()
			distance = np.linalg.norm(δ)
			
			   # Renormalize the pertubed vector (system2)
			x_perturbed = np.copy(system1.get_state() + (ε / distance) * δ)
			system2.set_state(x_perturbed)
	
			# Accumulate logarithmic growth
			log_sum += np.log(distance / ε)

			# Advance time
			total_time += dt
   
			# Progress updates
			progress = total_time / seconds * 100
			if progress >= next_percent_to_print:
				print(f"Wolf Algorithm progress: {int(progress)}%")
				next_percent_to_print += 5

		# compute Lyapunov exponent λ_max
		λ_max = log_sum / total_time
		print(f"Lyapunov exponent λ: {λ_max}")
		return λ_max

	"""
	Calculates all Lyapunov exponent λ for chaotic systems.
	Takes in initialized systems 1 and 2 for adding the small difference ε. This 
	will then run the simulations and calculate the difference in trajectories
	using each system.trajectory values. These values are orthonormalized using
	Gram-Schmidt orthogonalization and returned as an array
	 """
	def calculate_lyapunov(seconds, system, solver, ε=1e-8, dt=1e-3):
		def combined_dynamics(state):
			x = state[:3]
			Q = state[3:].reshape((3, 3))

			dxdt = system.compute_derivatives(x)
			J = system.jacobian(x)
			dQdt = J @ Q

			return np.concatenate([dxdt, dQdt.flatten()])
		λs = np.zeros(3)
		log_sums = np.zeros(3)
		# Gives the identity matrix with epsilon in the rows
		Q = np.eye(3) * ε
  
		state = np.concatenate([system.state, Q.flatten()])

		next_percent_to_print = 5 # First print at 5%

		total_time = 0
		n_steps = round(seconds / dt)
		# For each timestep in the tital number of timesteps:
		for step in range(n_steps):
			# Step with the system
			state = solver.step(combined_dynamics, state, dt)
			total_time += dt
   
			x = state[:3]
			Q = state[3:].reshape((3,3))
			
			# QR decomposition
			Q, R = np.linalg.qr(Q)
   
			# Accumulate logs
			diag_R = np.abs(np.diag(R))
			diag_R[diag_R < 1e-16] = 1e-16
			log_sums += np.log(diag_R)

			# rebuild state vector
			state = np.concatenate([x, Q.flatten()])
   
			# Progress updates
			progress = total_time / seconds * 100
			if progress >= next_percent_to_print:
				print(f"Bennetin Algorithm progress: {int(progress)}%")
				next_percent_to_print += 5
   
		λs = log_sums / total_time
  
		print(f"\n__Lyapunov Exponents__")
		for i, λ in enumerate(λs):
			print(f"λ{i}: {λ}")
   
		# Compute Kaplan Yorke dimension
		kaplan_yorke = 2 + (λs[0] / np.abs(λs[2]))
		print(f"\nKaplan Yorke dimension: {kaplan_yorke}")
		return λs, kaplan_yorke


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--method", type=str, default="DormandPrince45")
	parser.add_argument("--system", type=str, default="LorenzAttractor")
	parser.add_argument("--seconds", type=float, default=200) # We can either use seconds or total timesteps
	parser.add_argument("--dt", type=float, default=1e-3)
	parser.add_argument("--x", type=float, default=0.0)
	parser.add_argument("--y", type=float, default=1.0)	
	parser.add_argument("--z", type=float, default=0.0)
	parser.add_argument("--largest", action='store_true')

	parser.add_argument("--epsilon", type=float, default=1e-8)
	parser.add_argument("--tol", type=float, default=1e-4) # Tweak the solver tolerance for different chaotic systems
	args = parser.parse_args()
 
	adaptive = False # Fixes the timestep for RK methods, with no adaptive step logic. This is required for 
					 # the implemented Lyapunov solution
	  
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
	else:
		method_class = METHODS[args.method]
		solver = RungeKutta(method_class, adaptive)
	print(f"Using steplength {dt}")
 
	# Initialize the base system
	initial_state = np.array([args.x, args.y, args.z], dtype=float)
	if args.system == "LorenzAttractor":
		# Define the system used
		system1 = LorenzAttractor(initial_state=initial_state)
	elif args.system == "RosslerAttractor":
		system1 = RosslerAttractor(initial_state=initial_state)
	elif args.system == "ChuaCircuit":
		system1 = ChuaCircuit(initial_state=initial_state)
	elif args.system == "AizawaAttractor":
		system1 = AizawaAttractor(initial_state=initial_state)
	else:
		print("Incorrect system used. Exitting")
		exit(1)
  
	# Random unit vector in phase space
	v = np.random.randn(len(initial_state))
	v /= np.linalg.norm(v)
	perturbation = args.epsilon * v
	perturbed_state = initial_state + perturbation
	
	if args.system == "LorenzAttractor":
		system2 = LorenzAttractor(initial_state=perturbed_state)
	elif args.system == "RosslerAttractor":
		system2 = RosslerAttractor(initial_state=perturbed_state)
	elif args.system == "ChuaCircuit":
		system2 = ChuaCircuit(initial_state=perturbed_state)
	elif args.system == "AizawaAttractor":
		system2 = AizawaAttractor(initial_state=perturbed_state)
  
	if args.largest:
		print("Calculating only the Largest Lyapunov exponent.\n")
		λ_max = Lyapunov.calculate_largest_lyapunov(seconds=args.seconds, system1=system1, system2=system2, solver=solver, ε=args.epsilon, dt=args.dt)
	else:
		if args.system == "LorenzAttractor":
			λs, kaplan_yorke = Lyapunov.calculate_lyapunov(seconds=args.seconds, system=system1, solver=solver, ε=args.epsilon, dt=args.dt)
 