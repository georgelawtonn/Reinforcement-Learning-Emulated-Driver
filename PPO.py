from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from car_racing import CarRacing
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Plotting global variables
trainingRewards = []
speed_over_time = []
previous_position = None
previous_time = None


# Custom callback function for plotting
class TensorboardCallback(BaseCallback):
	def __init__(self, verbose=0):
		super(TensorboardCallback, self).__init__(verbose)

	def _on_step(self) -> bool:
		global previous_position, previous_time
		current_position = (
			self.training_env.get_attr('previous_x')[0],  # Extracting the actual values from the list
			self.training_env.get_attr('previous_y')[0]
		)
		current_time = self.num_timesteps  # Assuming num_timesteps gives the current time step

		if previous_position is not None and previous_time is not None:
			# Ensure positions are converted to numerical values before calculations
			prev_x, prev_y = previous_position
			curr_x, curr_y = current_position

			# Calculate speed based on change in position and time elapsed
			delta_position = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
			delta_time = current_time - previous_time
			speed = delta_position / delta_time if delta_time != 0 else 0.0
			speed_over_time.append(speed)

		# Update previous position and time for the next step
		previous_position = current_position
		previous_time = current_time
		if self.training_env.get_attr('lastEpisodeVal')[0] != 0:
			episode_info = self.training_env.get_attr('lastEpisodeVal')[0]
			trainingRewards.append(episode_info)
		return True


class PPOManager:
	# __init__ function
	def __init__(self):
		self.modelPPO = None
		self.defaultRender = "state_pixels"
		self.defaultObstacles = True
		self.defaultFile = "./ppo/"
		self.vec_env = None

	# Setup/Load for the model
	def setupModel(self, render, obstacle, fileLocation=None):
		# Setting up a base environment
		env = CarRacing(render_mode=render, continuous=False, domain_randomize=False, obstacles=obstacle)

		# Stack frames to provide the agent with a 2-frame history for understanding motion
		self.vec_env = DummyVecEnv([lambda: env])
		self.vec_env = VecFrameStack(self.vec_env, n_stack=2)

		# If the model is not passed in, a new model is initialized
		if fileLocation is None:
			self.modelPPO = PPO("CnnPolicy", self.vec_env)
		else:
			self.modelPPO = PPO.load(fileLocation)
			self.modelPPO.set_env(self.vec_env)

	# Training function
	def train(self, iterations, save_interval, logging, save_location="./ppo/"):
		if self.modelPPO is None:
			self.setupModel(self.defaultRender, self.defaultObstacles)

		if logging:
			callback = TensorboardCallback()
			for t in tqdm(range(iterations)):
				# Trains the model for 5000 timesteps
				self.modelPPO.learn(total_timesteps=5000, callback=callback)

				# Saves the model at a certain interval
				if t % save_interval == 0:
					self.modelPPO.save(f"{self.defaultFile}{save_location}_{t}.zip")

			self.modelPPO.save(f"{self.defaultFile}{save_location}_finalSave.zip")
			self.plotResults()
		else:
			for t in tqdm(range(iterations)):
				# Trains the model for 5000 timesteps
				self.modelPPO.learn(total_timesteps=5000)

				# Saves the model at a certain interval
				if t % save_interval == 0:
					self.modelPPO.save(f"{self.defaultFile}{save_location}_{t}.zip")

			self.modelPPO.save(f"{self.defaultFile}{save_location}_finalSave.zip")

	# Plotter function
	def plotResults(self):
		# Plotting
		plt.plot(trainingRewards)
		plt.xlabel('Episode')
		plt.ylabel('Reward')
		plt.title('Reward To Episode')
		plt.show()

		plt.plot(speed_over_time)
		plt.xlabel('Time steps')
		plt.ylabel('Speed')
		plt.title('Car Speed Over Time')
		plt.show()

	# Runs the model
	def run(self, steps):
		obs = self.vec_env.reset()
		for _ in range(steps):
			action, _ = self.modelPPO.predict(obs)
			obs, reward, done, info = self.vec_env.step(action)
			self.vec_env.render()
			if done:
				obs = self.vec_env.reset()


# Initializing manager
ppo_manager = PPOManager()

# Parameters for setupModel:
# render = state_pixels (No rendering), human (Visible rendering)
# obstacles = Boolean (Turns obstacles on and off)
# fileLocation = String (Location of load file if loading is necessary)
ppo_manager.setupModel(render="human", obstacle=True)

# Parameters for train:
# iterations = Int (The number of times the model should be trained on 5000 timesteps)
# save_interval = Int (The number of times between saves)
# logging = Boolean (Whether or not graphs/plots should be displayed)
# save_location = String (Default: "./ppo/", the save location)
ppo_manager.train(iterations=2, save_interval=2, logging=True)

# If using a pretrained model run allows for you to run a model (After loading with setupModel)
# Parameters for run
# steps = Int (representation of the number of steps the model should be run for)
# ppo_manager.run(steps=10000)


# If looking to train a model similiar to those in our repository, do not specify a fileLocation on load and train for
# around 200 iterations, for without obstacles (More with obstacles). To load our current models just put fileLocation =
# *name of zip*

