import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
"""
The Visualize class is responsible for generating an image of the trajectory using matplotlib.
This is a static image with no animation capabilities.

This has one function for displaying the graph. Technically we could store the simulator
but this is not currently implemented
"""
class Visualize:
	def plot(self, steps, times, trajectory, method_name, dt=None):
		traject_arr = np.array(trajectory)
		x, y, z = traject_arr[:, 0], traject_arr[:, 1], traject_arr[:, 2]
		
		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
		
		if dt:
			label = f"Method: {method_name} | time: {round(times[-1], 2)}s | timesteps: {steps} | dt = {dt}"
		else:
			label = f"Method: {method_name} | time: {round(times[-1], 2)}s | timesteps: {steps}"
		
		ax.plot(x, y, z, lw=.2, label=label)
		ax.set_title(f"Lorenz Attractor")
		plt.legend()
		
		plt.show()
		

class Animate:
    
	def animate(self, steps, time, trajectory, method_name, sample_dt, interval):
     
     
		

		traject_arr = np.array(trajectory)

		x, y, z = traject_arr[:, 0], traject_arr[:, 1], traject_arr[:, 2]

		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
  
		# Set limits for better visuals
		ax.set_xlim(np.min(x), np.max(x))
		ax.set_ylim(np.min(y), np.max(y))
		ax.set_zlim(np.min(z), np.max(z))
  
		ax.set_title(f"Lorenz Attractor using {method_name} method")
  
		# Line trajectory 
		line, = ax.plot([],[],[],lw=.5)
  
		# Current position 
		point, = ax.plot([], [], [], 'ro', markersize=6)
  
		# Displays the time
		time_text = ax.text2D(.05, .90, '', transform=ax.transAxes)
  
		"""
		Draws the trajectory up to the current frame and updates the moving point.
		These frame updates are layered on top of each other to achieve the animation.
		"""
		def update(frame):
			# Draw the trajectory to the current frame
			line.set_data(x[:frame], y[:frame])
			line.set_3d_properties(z[:frame])
   
			# Update the moving point 
			point.set_data([x[frame]], [y[frame]])
			point.set_3d_properties([z[frame]])
   
			# Set the time 
			time_text.set_text(f"t = {frame * sample_dt:.2f}s")
			
			return line, point, time_text
  
		animation = FuncAnimation(
			fig,
			update,
			frames=len(x),
			interval=interval,
			blit=True
		)

		plt.show()
  