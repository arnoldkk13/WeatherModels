import numpy as np

"""
Animate houses the logic for animating the trajectory of the system using FuncAnimation. This is done
by layering frames on top of each other starting from the beginning trajectory and ending at the end.
"""
class Animate:
	"""
	Animate will draw the trajectory from the passed in trajectory as frames. This takes in a 
	"""
	def animate(self, steps, time, trajectory, method_name, sample_dt, system, update_method='linear', duration=150, save=False):
		import matplotlib 
		if save:
			matplotlib.use('Agg') 
		print(f'matplotlib backend: {matplotlib.get_backend()}')
		import matplotlib.pyplot as plt
		from matplotlib.animation import FuncAnimation

		traject_arr = np.array(trajectory)

		x, y, z = traject_arr[:, 0], traject_arr[:, 1], traject_arr[:, 2]
  
		TARGET_FPS = 60
		TARGET_DURATION_S = duration
		n_frames = TARGET_FPS * TARGET_DURATION_S  # 3600 frames for 60s at 60fps
		interval = 1000 // TARGET_FPS  # Fixed ~17ms per frame
  
		"""
		Returns a modified trajectory as a result of the time and length of the trajectory object for a smooth
		image. This can be done either linearly or exponentially (gives the appearance of speeding up)
		"""
		def set_trajectory_for_animation(trajectory, update_method):
			n = len(trajectory)
			if update_method == 'linear':
				indices = np.linspace(0, n - 1, n_frames, dtype=int)
			elif update_method == 'exponential':
				exp_vals = np.logspace(0, np.log10(n), n_frames)
				indices = np.clip(exp_vals - 1, 0, n - 1).astype(int)
			else:
				raise ValueError(f"Unknown update_method '{update_method}'. Expected 'linear' or 'exponential'.")
			return indices

		frame_indices = set_trajectory_for_animation(traject_arr, update_method)

		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
  
		# Set limits for better visuals
		ax.set_xlim(np.min(x), np.max(x))
		ax.set_ylim(np.min(y), np.max(y))
		ax.set_zlim(np.min(z), np.max(z))
  
		titles = {
			"LorenzAttractor":  "Lorenz Attractor",
			"RosslerAttractor": "Rossler Attractor",
			"ChuaCircuit":      "Chua's Circuit",
			"AizawaAttractor":  "Aizawa Attractor",
		}
		title = titles.get(system, system)
		ax.set_title(f"{title} using {method_name} method")
		ax.set_xlabel("X")
		ax.set_ylabel("Y")
		ax.set_zlabel("Z")
  
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
			idx = frame_indices[frame]
			# Draw the trajectory to the current frame
			line.set_data(x[:idx], y[:idx])
			line.set_3d_properties(z[:idx])
   
			# Update the moving point 
			point.set_data([x[idx]], [y[idx]])
			point.set_3d_properties([z[idx]])
   
			# Set the time 
			time_text.set_text(f"t = {idx * sample_dt:.2f}s")
			
			return line, point, time_text
  
		animation = FuncAnimation(
			fig,
			update,
			frames=n_frames,
			interval=interval,
			blit=True
		)
		if not save:
			plt.show()
		if save:
			from matplotlib.animation import FFMpegWriter

			writer = FFMpegWriter(fps=60)
			with writer.saving(fig, "output.mp4", dpi=100):
				for i in range(n_frames):
					update(i) 
					writer.grab_frame()

			animation.save(
				f'videos/{method_name}_animation_{time}s.mp4',
				writer=writer
			)
			print(f"Saved animation to videos/{method_name}_animation_{time}s.mp4")
		
  

"""
The Visualize class is responsible for generating an image of the trajectory using matplotlib.
This is a static image with no animation capabilities.

This has one function for displaying the graph. Technically we could store the simulator error
but this is not currently implemented
"""
class Visualize:
	def plot(self, steps, times, trajectory, method_name, system, dt=None):
		import matplotlib.pyplot as plt
		import matplotlib as mpl

		from mpl_toolkits.mplot3d.art3d import Line3DCollection

		traject_arr = np.array(trajectory)
		x, y, z = traject_arr[:, 0], traject_arr[:, 1], traject_arr[:, 2]
		
		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')

		points = np.array([x, y, z]).T.reshape(-1, 1, 3)
		segments = np.concatenate([points[:-1], points[1:]], axis=1)
  
		t = np.array(times)
		t_norm = (t - t[0]) / (t[-1] - t[0])
  
		cmap = plt.cm.plasma
		norm = mpl.colors.Normalize(vmin=t_norm.min(), vmax=t_norm.max())
  
		lc = Line3DCollection(segments, cmap=cmap, norm=norm)
		lc.set_array(t_norm[:-1])   # one color per segment
		lc.set_linewidth(0.2)
  
		ax.add_collection(lc)
  
		ax.auto_scale_xyz(x, y, z)

		
		if dt:
			label = f"Method: {method_name} | time: {round(times[-1], 2)}s | timesteps: {steps} | dt = {dt}"
		else:
			label = f"Method: {method_name} | time: {round(times[-1], 2)}s | timesteps: {steps}"
  
		# Create an invisible line for the legend
		ax.plot([], [], [], color=cmap(0.8), label=label)
		ax.set_xlabel("X")
		ax.set_ylabel("Y")
		ax.set_zlabel("Z")
		
		if system == "LorenzAttractor":
			ax.set_title(f"Lorenz Attractor using {method_name} method")
		elif system == "RosslerAttractor":
			ax.set_title(f"Rossler Attractor using {method_name} method")
		elif system == "ChuaCircuit":
			ax.set_title(f"Chua's Circuit using {method_name} method")
		elif system == "AizawaAttractor":
			ax.set_title(f"Aizawa Attractor using {method_name} method")
   
		cbar = fig.colorbar(lc, ax=ax, pad=0.1)
		cbar.set_label("Normalized Time")

		plt.legend()
		
		plt.show()
		
