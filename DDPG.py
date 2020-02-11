import pandas as pd
import numpy as np
import gym
import multiprocessing
import pickle
import tensorflow as tf
import time
seed = 0
tf.random.set_seed(seed)
np.random.seed(seed)

#### Helper Functions
def neural_network(input_shape,activation,output_activation,output_shape):
	model = tf.keras.Sequential([tf.keras.layers.Dense(64,activation = activation,input_shape = input_shape),
		tf.keras.layers.Dense(64,activation = activation),
		#tf.keras.layers.Dense(64,activation = activation),
		tf.keras.layers.Dense(output_shape,activation = output_activation)])
	return model

#### The agent starts here 
class Agent():
	def __init__(self):
		self.learning_rate = 0.0001
		self.max_episode_length = 200
		self.gamma = 0.99
		self.polyak = 0.005
		self.epochs = 300

		self.obs_space = 3
		self.acs_space = 1

		#### Building actor and critic
		self.Q_network = neural_network([self.obs_space + self.acs_space],tf.nn.relu,None,1)
		self.policy = neural_network([self.obs_space],tf.nn.relu,tf.tanh,self.acs_space)
		#self.policy = tf.keras.models.load_model('C:/Users/vlpap/Desktop/policy.h5')

		#### Building target networks ######
		self.target_Q_network = tf.keras.models.clone_model(self.Q_network)
		self.target_Q_network.build()
		self.target_Q_network.set_weights(self.Q_network.get_weights())
		self.target_policy = tf.keras.models.clone_model(self.policy)
		self.target_policy.build()
		self.target_policy.set_weights(self.policy.get_weights())

		#### Initializing replay buffer
		self.buffer = {"state":[],"rewards":[],"actions":[], "next_state":[]}
		self.buffer_length = 0
		self.max_buffer_size = 100000
		self.batch_size = 128

	
	def get_exploration_action(self,ob,model):
		action = model.predict(ob)
		random = tf.random.normal(tf.shape(action))
		action = action * 2 + 0.1 * random
		action = np.array(tf.clip_by_value(action, -2, 2))
		return action

	##### This method selects batch size in order, Next implement the function such that the batch pairs picked are completely random
	##### and uncorrelated!
	def select_batch(self):

		if self.buffer_length <= self.batch_size:
			index = np.random.randint(0,self.buffer_length,self.buffer_length)

		else:
			index = np.random.randint(0,self.buffer_length,self.batch_size)
		
		batch = (np.array(self.buffer['state'])[index], np.array(self.buffer['rewards'])[index], np.array(self.buffer['actions'])[index],\
			np.array(self.buffer['next_state'])[index])
		return batch


	def calc_target_Q(self,rews,obs_next):
		target = []
		l = len(obs_next)
		for i in range(l):
			action = self.target_policy.predict(obs_next[i].reshape(1,-1)) * 2
			sa_pair = np.concatenate([obs_next[i], action[0]]).reshape(1,-1)
			q_next = self.target_Q_network.predict(sa_pair)[0][0]
			tar = rews[i] + self.gamma * q_next
			target.append(tar)

		return target


	def update(self,target,acs,obs):
		#### Critic Update
		critic_param = self.Q_network.trainable_variables
		with tf.GradientTape() as tape:
			tape.watch(critic_param)
			#obs = np.array(obs)
			#acs = np.array(acs)
			sa_pair = np.concatenate([obs,acs], axis = 1)
			target = np.array(target).reshape(-1,1)
			v_loss = tf.reduce_mean((target - self.Q_network(sa_pair))**2)
		grad = tape.gradient(v_loss, critic_param)
		optimizer = tf.keras.optimizers.Adam(0.001)
		optimizer.apply_gradients(zip(grad, critic_param))


		#### Policy Update
		actor_param = self.policy.trainable_variables
		with tf.GradientTape() as tape:
			tape.watch(actor_param)
			policy_action = self.policy(obs) * 2
			sa_pair = tf.concat([obs, policy_action], axis = 1)
			pi_loss = -tf.reduce_mean(self.Q_network(sa_pair))
			grad = tape.gradient(pi_loss, actor_param)
		optimizer = tf.keras.optimizers.Adam(0.0001)
		optimizer.apply_gradients(zip(grad, actor_param))

		#print("Q Function loss ",v_loss)
		#print("Pi Loss ",pi_loss)



	def target_update(self):
		
		# Updating critic target network
		l = len(self.Q_network.trainable_variables)
		for i in range(l):
			self.target_Q_network.trainable_variables[i].assign(self.polyak * self.Q_network.trainable_variables[i] + (1 - self.polyak) * self.target_Q_network.trainable_variables[i])

		#Updating policy target network
		l = len(self.policy.trainable_variables)
		for i in range(l):
			self.target_policy.trainable_variables[i].assign(self.polyak * self.policy.trainable_variables[i] + (1 - self.polyak) * self.target_policy.trainable_variables[i])


	def final(self):
		#### Environment Initialization
		env = gym.make('Pendulum-v0')

		#### Run the environment N number of times
		#### Occasionally check the deterministic policy performance
		for i in range(self.epochs):
			ob = env.reset()
			print("######## EPISODE NUMBER ",i," ########")
			t1 = time.time()
			for j in range(self.max_episode_length):
				self.buffer['state'].append(ob)
				ob = np.array(ob).reshape(1,-1)
				ac = self.get_exploration_action(ob,self.policy)
				ac = ac[0]
				ob, rew, done, _ = env.step(ac)
				#print("THE ACTION WAS ", ac)
				self.buffer['next_state'].append(ob)
				self.buffer['rewards'].append(rew)
				self.buffer['actions'].append(ac)
				self.buffer_length += 1

				#### Select randomly a batch to train on
				batch = self.select_batch()
				obs, rews, acs, obs_next = batch

				#### Calculate target yi to update critic network
				target = self.calc_target_Q(rews, obs_next)

				#### Update after getting the targets
				self.update(target, acs, obs)

				#### Update the target networks
				self.target_update()
				if done:
					break
			t2 = time.time()
			print("TIME TAKEN FOR THIS EPISODE ", t2-t1)

			
				#### Ocassionally checking the performance of the deterministic policy
			ob = env.reset()
			total_reward = 0
			for k in range(self.max_episode_length):
				#env.render()
				ob = np.array(ob).reshape(1,-1)
				ac = self.policy.predict(ob) * 2 
				np.clip(ac,-2,2)
				ac = ac[0]
				ob, rew, done, _ = env.step(ac)
				total_reward += rew
				if done:
					break
			print(" The performance of the deterministic policy at ",i+1," is ",total_reward)
			
			if (i + 1) % 5 == 0:
				self.policy.save('')
				self.target_policy.save('')
				self.Q_network.save('')
				self.target_Q_network.save('')



agent = Agent()
agent.final()


