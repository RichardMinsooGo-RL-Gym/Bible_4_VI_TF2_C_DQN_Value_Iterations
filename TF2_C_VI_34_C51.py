# IMPORTING LIBRARIES

import sys
IN_COLAB = "google.colab" in sys.modules

import random
import gym
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax
from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model
from collections import deque

from IPython.display import clear_output
import math

tf.keras.backend.set_floatx('float64')

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
    
    def put(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])
    
    def sample(self):
        sample = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, done = map(
            np.asarray, zip(*sample))
        states = np.array(states).reshape(batch_size, -1)
        next_states = np.array(next_states).reshape(batch_size, -1)
        return states, actions, rewards, next_states, done
    
    def size(self):
        return len(self.memory)

# Neural Network Model Defined at Here.
class Network(Model):
    def __init__(self, state_size: int, action_size: int, 
    z):
        """Initialization."""
        super(Network, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.atoms = atoms
        self.z = z

        # self.opt = tf.keras.optimizers.Adam(lr)
        self.model = self.create_model()
    
    def create_model(self):
        input_state = Input((self.state_size,))
        h1 = Dense(64, activation='relu')(input_state)
        h2 = Dense(64, activation='relu')(h1)
        outputs = []
        for _ in range(self.action_size):
            outputs.append(Dense(self.atoms, activation='softmax')(h2))
        return tf.keras.Model(input_state, outputs)
    
    def predict(self, state):
        return self.model.predict(state)

class DQNAgent:
    def __init__(
        self, 
        env: gym.Env,
        batch_size: int,
        target_update: int,
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            epsilon_decay (float): step size to decrease epsilon
            lr (float): learning rate
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            gamma (float): discount factor
        """
        
        # CREATING THE Q-Network
        self.env = env
        self.env.seed(1234)  
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        self.batch_size = batch_size
        # hyper parameters
        memory_size = 10000
        self.v_max = v_max
        self.v_min = v_min
        self.atoms = atoms
        self.delta_z = float(self.v_max - self.v_min) / (self.atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.atoms)]
        self.lr = 0.001
        self.target_update = target_update
        self.gamma = 0.99    # discount rate
        
        # create main model and target model
        self.dqn = Network(self.state_size, self.action_size, self.z
                          )
        self.dqn_target = Network( self.state_size, self.action_size, self.z
                          )
        self.train_start = 1000
        
        self.optimizers = optimizers.Adam(lr=self.lr, )
        self.criterion = tf.keras.losses.CategoricalCrossentropy()
        
        self.memory = ReplayBuffer(capacity=memory_size)
        self.Soft_Update = False # use soft parameter update

        self.TAU = 0.1 # target network soft update hyperparameter
        
        self._target_hard_update()
        
    # EXPLORATION VS EXPLOITATION
    def get_action(self, state, epsilon):
        state = np.reshape(state, [1, self.state_size])
        # 3. Choose an action a in the current world state (s)
        # If this number < greater than epsilon doing a random choice --> exploration
        if np.random.rand() <= epsilon:
            action = np.random.choice(self.action_size)

        ## Else --> exploitation (taking the biggest Q value for this state)
        else:
            action = self.get_optimal_action(state)

        return action
    
    def get_optimal_action(self, state):
        z = self.dqn.predict(state)
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        return np.argmax(q)
    
    # UPDATING THE Q-VALUE
    def train_step(self):
        states, actions, rewards, next_states, dones = self.memory.sample()
        z = self.dqn.predict(next_states)
        z_ = self.dqn_target.predict(next_states)
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        q = q.reshape((self.batch_size, self.action_size), order='F')
        next_actions = np.argmax(q, axis=1)
        m_prob = [np.zeros((self.batch_size, self.atoms))
                  for _ in range(self.action_size)]
        for i in range(self.batch_size):
            if dones[i]:
                T_zj = min(self.v_max, max(self.v_min, rewards[i]))
                b_j  = (T_zj - self.v_min) / self.delta_z
                l, u = math.floor(b_j), math.ceil(b_j)
                m_prob[actions[i]][i][int(l)] += (u - b_j)
                m_prob[actions[i]][i][int(u)] += (b_j - l)
            else:
                for j in range(self.atoms):
                    T_zj = min(self.v_max, max(
                        self.v_min, rewards[i] + self.gamma * self.z[j]))
                    b_j = (T_zj - self.v_min) / self.delta_z
                    l, u = math.floor(b_j), math.ceil(b_j)
                    m_prob[actions[i]][i][int(
                        l)] += z_[next_actions[i]][i][j] * (u - b_j)
                    m_prob[actions[i]][i][int(
                        u)] += z_[next_actions[i]][i][j] * (b_j - l)
        # self.dqn.train(states, m_prob)
        # def train(self, states, targets):
        m_prob = tf.stop_gradient(m_prob)
        dqn_variable = self.dqn.model.trainable_variables
        with tf.GradientTape() as tape:
            main_value = self.dqn.model(states)
            loss = self.criterion(m_prob, main_value)
            
        dqn_grads = tape.gradient(loss, dqn_variable)
        self.optimizers.apply_gradients(zip(dqn_grads, dqn_variable))
        
    # after some time interval update the target model to be same with model
    def _target_hard_update(self):
        if not self.Soft_Update:
            weights = self.dqn.model.get_weights()
            self.dqn_target.model.set_weights(weights)
            return
        if self.Soft_Update:
            q_model_theta = self.dqn.model.get_weights()
            dqn_target_theta = self.dqn_target.model.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, dqn_target_theta):
                target_weight = target_weight * (1-self.TAU) + q_weight * self.TAU
                dqn_target_theta[counter] = target_weight
                counter += 1
            self.dqn_target.set_weights(dqn_target_theta)
    
    def load(self, name):
        self.dqn = load_model(name)

    def save(self, name):
        self.dqn.save(name)
    
# CREATING THE ENVIRONMENT
env_name = "CartPole-v0"
env = gym.make(env_name)

# parameters
target_update = 20


# INITIALIZING THE Q-PARAMETERS
hidden_size = 128
max_episodes = 200  # Set total number of episodes to train agent on.
batch_size = 64

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.025            # Exponential decay rate for exploration prob

atoms = 51
v_min = -10.
v_max = 10.

# train
agent = DQNAgent(
    env, 
#     memory_size, 
    batch_size, 
    target_update, 
#     epsilon_decay,
)

if __name__ == "__main__":
    
    update_cnt    = 0
    # TRAINING LOOP
    #List to contain all the rewards of all the episodes given to the agent
    scores = []
    
    # EACH EPISODE    
    for episode in range(max_episodes):
        ## Reset environment and get first new observation
        state = agent.env.reset()
        episode_reward = 0
        done = False  # has the enviroment finished?
        
            
        # EACH TIME STEP    
        while not done:
        # for step in range(max_steps):  # step index, maximum step is 200
        
            # 3.4.1 EXPLORATION VS EXPLOITATION
            # Take the action (a) and observe the outcome state(s') and reward (r)
            action = agent.get_action(state, epsilon)
            
            # 2.7.2 TAKING ACTION
            next_state, reward, done, _ = agent.env.step(action)
            agent.memory.put(state, action, reward, next_state, done)
            
            # Our new state is state
            state = next_state

            episode_reward += reward

            # if episode ends
            if done:
                scores.append(episode_reward)
                print("Episode: {}/{}, Score: {}, Epsilon: {:.4}".format(episode+1, max_episodes, episode_reward, epsilon)) 
                break
            # if training is ready
            if agent.memory.size() >= agent.batch_size:
                # 3.4.2 UPDATING THE Q-VALUE
                agent.train_step()
                update_cnt += 1
            
                # if hard update is needed
                if update_cnt % agent.target_update == 0:
                    agent._target_hard_update()
            
        # 2.8 EXPLORATION RATE DECAY
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 

        