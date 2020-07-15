#!/usr/bin/env python
# coding: utf-8

# # Building an Agent to Play Atari games using Deep Q Network

# First we import all the necessary libraries </font> 
# 

# In[1]:


import csv
import os

import numpy as np
import gym
import matplotlib.pyplot as plt
from skimage.transform import resize
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(tf.__version__)

# from tensorflow.contrib.layers import flatten, conv2d, fully_connected
from tensorflow.compat.v1.layers import flatten, conv2d, dense

from collections import deque, Counter
import random
from datetime import datetime


# Now we define a function called preprocess_observation for preprocessing our input game screen. We reduce the image size
# and convert the image into greyscale.

# In[2]:


color = np.array([210, 164, 74]).mean()

def preprocess_observation(obs):

    # Crop and resize the image
    img = obs[1:176:2, ::2]

    # Convert the image to greyscale
    img = img.mean(axis=2)

    # Improve image contrast
    img[img==color] = 0

    # Next we normalize the image from -1 to +1
    img = (img - 128) / 128 - 1

    return img.reshape(88,80,1)


#  Let us initialize our gym environment

# In[3]:


env = gym.make("Breakout-v0") #gym.make("MsPacman-v0")
n_outputs = env.action_space.n

print("n_outputs: ", n_outputs)
print("action_space:", env.action_space)
print("observation_space:", env.observation_space)


# In[4]:


filename = "original-512" #"my_fix-512"

if filename.startswith('original'):
    print("original")
else:
    print("my_fix")


# Okay, Now we define a function called q_network for building our Q network. We input the game state
# to the Q network and get the Q values for all the actions in that state. <br><br>
# We build Q network with three convolutional layers with same padding followed by a fully connected layer. 

# In[5]:


tf.reset_default_graph()

def q_network(X, name_scope):
    
    # Initialize layers
    # initializer = tf.contrib.layers.variance_scaling_initializer()
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    with tf.variable_scope(name_scope) as scope: 

        # initialize the convolutional layers
        #layer_1 = conv2d(X, num_outputs=32, kernel_size=(8,8), stride=4, padding='SAME', weights_initializer=initializer) 
        layer_1 = conv2d(X, filters=32, kernel_size=(8,8), strides=4, padding='SAME')
        tf.summary.histogram('layer_1',layer_1)
        
        #layer_2 = conv2d(layer_1, num_outputs=64, kernel_size=(4,4), stride=2, padding='SAME', weights_initializer=initializer)
        layer_2 = conv2d(layer_1, filters=64, kernel_size=(4,4), strides=2, padding='SAME')
        tf.summary.histogram('layer_2',layer_2)
        
        #layer_3 = conv2d(layer_2, num_outputs=64, kernel_size=(3,3), stride=1, padding='SAME', weights_initializer=initializer)
        layer_3 = conv2d(layer_2, filters=64, kernel_size=(3,3), strides=1, padding='SAME')
        tf.summary.histogram('layer_3',layer_3)
        
        # Flatten the result of layer_3 before feeding to the fully connected layer
        flat = flatten(layer_3)

        #fc = fully_connected(flat, num_outputs=128, weights_initializer=initializer)
        #fc = dense(flat, units=128)
        fc = dense(flat, units=512) #GT
        tf.summary.histogram('fc',fc)
        
        #output = fully_connected(fc, num_outputs=n_outputs, activation_fn=None, weights_initializer=initializer)
        output = dense(fc, units=n_outputs, activation=None)
        tf.summary.histogram('output',output)
        
        # Vars will store the parameters of the network such as weights
        vars = {v.name[len(scope.name):]: v for v in tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)} 
        return vars, output


# Next we define a function called epsilon_greedy for performing epsilon greedy policy. In epsilon greedy policy we either select the best action with probability 1 - epsilon or a random action with
# probability epsilon.
# 
# We use decaying epsilon greedy policy where value of epsilon will be decaying over time as we don't want to explore
# forever. So over time our policy will be exploiting only good actions.

# In[6]:


epsilon = 0.5
eps_min = 0.05
eps_max = 1.0
eps_decay_steps = 500000

def epsilon_greedy(action, step):
    p = np.random.random(1).squeeze()
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    # print("epsilon: ", epsilon)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        return action

epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay_steps = 500000
# The epsilon decay schedule
epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

def epsilon_greedy_gt(action, step):
    epsilon = epsilons[min(step, epsilon_decay_steps-1)]
    #print("epsilon: ", epsilon)
    
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        return action


# In[7]:


"""
import numpy as np

rand = np.random.rand()
print(rand)
epsilon = 0.9

if rand < epsilon:
    print("RANDOM")
else:
    print("Selected ACTION")
"""


# Now, we initialize our experience replay buffer of length 20000 which holds the experience.
# 
# We store all the agent's experience i.e (state, action, rewards) in the experience replay buffer
# and  we sample from this minibatch of experience for training the network.

# In[8]:


#buffer_len = 20000
buffer_len = 500000 # GT
exp_buffer = deque(maxlen=buffer_len)


# Next, we define a function called sample_memories for sampling experiences from the memory. Batch size is the number of experience sampled
# from the memory.
# 

# In[9]:


def sample_memories(batch_size):
    perm_batch = np.random.permutation(len(exp_buffer))[:batch_size]
    mem = np.array(exp_buffer)[perm_batch]
    return mem[:,0], mem[:,1], mem[:,2], mem[:,3], mem[:,4] # obs, action, next_obs, reward, done


# Now we define our network hyperparameters,

# In[10]:


num_episodes = 10000 #800
#batch_size = 48
batch_size = 32 #GT
#input_shape = (None, 88, 80, 1)
#input_shape = (None, 84, 84, 1) #GT
input_shape = (None, 84, 84, 4) #GT
learning_rate = 0.001
#X_shape = (None, 88, 80, 1)
#X_shape = (None, 84, 84, 1) #GT
X_shape = (None, 84, 84, 4) #GT
#discount_factor = 0.97
discount_factor = 0.99 #GT

global_step = 0
#copy_steps = 100
copy_steps = 10000 #GT
#steps_train = 4
steps_train = 1 # GT
#start_steps = 2000
start_steps = 50000 #GT


# In[11]:


logdir = filename

os.makedirs(logdir, exist_ok=True)

tf.reset_default_graph()

# Now we define the placeholder for our input i.e game state
X = tf.placeholder(tf.float32, shape=X_shape)

# we define a boolean called in_training_model to toggle the training
in_training_mode = tf.placeholder(tf.bool)


#  Now let us build our primary and target Q network

# In[12]:


# we build our Q network, which takes the input X and generates Q values for all the actions in the state
mainQ, mainQ_outputs = q_network(X, 'mainQ')

# similarly we build our target Q network
targetQ, targetQ_outputs = q_network(X, 'targetQ')


# In[13]:


# define the placeholder for our action values
X_action = tf.placeholder(tf.int32, shape=(None,))
if filename.startswith('original'):
    Q_action = tf.reduce_sum(targetQ_outputs * tf.one_hot(X_action, n_outputs), axis=-1, keep_dims=True)
else:
    Q_action = tf.reduce_sum(mainQ_outputs * tf.one_hot(X_action, n_outputs), axis=-1, keep_dims=True) #GT

print("Q_action: ", Q_action)


# Copy the primary Q network parameters to the target  Q network

# In[14]:


if filename.startswith('original'):
    copy_op = [tf.assign(main_name, targetQ[var_name]) for var_name, main_name in mainQ.items()]
    copy_target_to_main = tf.group(*copy_op)
else:
    #GT
    #tf.compat.v1.assign(ref, value): outputs a Tensor that holds the new value of ref after the value has been assigned
    copy_op = [tf.assign(target_name, mainQ[var_name]) for var_name, target_name in targetQ.items()]
    copy_target_to_main = tf.group(*copy_op)


# Compute and optimize loss using gradient descent optimizer

# In[15]:


# define a placeholder for our output i.e action
y = tf.placeholder(tf.float32, shape=(None,1))

# now we calculate the loss which is the difference between actual value and predicted value
loss = tf.reduce_mean(tf.square(y - Q_action))

# we use adam optimizer for minimizing the loss
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

loss_summary = tf.summary.scalar('LOSS', loss)
merge_summary = tf.summary.merge_all()
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

checkpoint_path = logdir
saver = tf.train.Saver()
# latest_checkpoint = checkpoint_path #GT: load model
# if latest_checkpoint:
#     print("Loading model checkpoint {}...\n".format(latest_checkpoint))
#     saver.restore(sess, latest_checkpoint)


# In[16]:


def preprocess_observation_gt(obs):

    # Crop
    img = obs[34:34+160]
    
    # Convert the image to greyscale
    img = img.mean(axis=2)

    # Resize
    img = resize(img, (84, 84))

    return img.reshape(84, 84, 1)


# In[17]:


# For Testing....

"""
obs = env.reset()

print("obs: ", obs.shape)
plt.imshow(obs)
plt.colorbar()
plt.show()

# get the preprocessed game screen
#obs = preprocess_observation(obs)
obs = preprocess_observation_gt(obs)

print("obs: ", obs.shape)
plt.imshow(obs.reshape((obs.shape[0], obs.shape[1])))
plt.colorbar()
plt.show()
"""


# In[18]:


class StateProcessor():
    """
    Processes a raw Atari images. Resizes it and converts it to grayscale.
    """
    def __init__(self):
        # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State

        Returns:
            A processed [84, 84] state representing grayscale values.
        """
        return sess.run(self.output, { self.input_state: state })


# In[19]:


# For Testing....

"""
tf.reset_default_graph()
global_step = tf.Variable(0, name="global_step", trainable=False)

sp = StateProcessor()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # Example observation batch
    state = env.reset()
    
    print("ORIGINAL STATE")
    print("state: ", state.shape)
    plt.imshow(state)
    plt.colorbar()
    plt.show()
    
    print("PROCESSED STATE")
    state_p = sp.process(sess, state)
    print("state_p: ", state_p.shape)
    print("state_p[0][0]: ", state_p[0][0])
    plt.imshow(state_p)
    plt.colorbar()
    plt.show()
        
    print("SAME 4 PIXCELS IN 3rd DIMENSION")
    state = np.stack([state_p] * 4, axis=2)
    print("state: ", state.shape)
    print("state[0][0]: ", state[0][0])
    plt.imshow(state)
    plt.colorbar()
    plt.show()
    
    next_state, reward, done, _ = env.step(2)
    next_state = sp.process(sess, next_state)
    print("state[:,:,1:]: ", state[:,:,1:].shape)
    print("np.expand_dims(next_state, 2): ", np.expand_dims(next_state, 2).shape)
    next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
    
    print("NEXT STATE - 4 PIXCELS IN 3rd DIMENSION: 3 FROM LAST SATE & 1 FROM NEXT STATE.")
    print("TO HIGHLIGHT THE CHANGE")
    print("next_state: ", next_state.shape)
    plt.imshow(next_state)
    plt.colorbar()
    plt.show()
"""


# Now we start the tensorflow session and run the model,

# In[ ]:


state_processor = StateProcessor()

with open(logdir+os.sep+filename+'.csv','w') as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(['episode', 'reward'])
    
with tf.Session() as sess:
    init.run()
    
    all_episodic_reward = []
    # for each episode
    for i in range(num_episodes):
        
        # Save the current checkpoint
        saver.save(tf.get_default_session(), logdir+os.sep+"model"+os.sep+filename) #GT
        
        done = False
        obs = env.reset()
        epoch = 0
        episodic_reward = 0
        actions_counter = Counter() 
        episodic_loss = []
        
        obs = state_processor.process(sess, obs) #GT
        obs = np.stack([obs] * 4, axis=2) #GT

        # while the state is not the terminal state
        while not done:

            #env.render()
        
            # get the preprocessed game screen
            #obs = preprocess_observation(obs)
            #obs = preprocess_observation_gt(obs) #GT
            
            #plt.imshow(obs.reshape((obs.shape[0], obs.shape[1])))
            #plt.colorbar()
            #plt.show()

            # feed the game screen and get the Q values for each action
            actions = mainQ_outputs.eval(feed_dict={X:[obs], in_training_mode:False})

            # get the action
            action = np.argmax(actions, axis=-1)
            actions_counter[str(action)] += 1 

            # select the action using epsilon greedy policy
            action = epsilon_greedy(action, global_step)
            #action = epsilon_greedy_gt(action, global_step) #GT
            
            # now perform the action and move to the next state, next_obs, receive reward
            next_obs, reward, done, _ = env.step(action)

            # Store this transistion as an experience in the replay buffer
            #exp_buffer.append([obs, action, preprocess_observation(next_obs), reward, done])
            #exp_buffer.append([obs, action, preprocess_observation_gt(next_obs), reward, done]) #GT
            next_obs = state_processor.process(sess, next_obs) #GT
            next_obs = np.append(obs[:,:,1:], np.expand_dims(next_obs, 2), axis=2) # GT            
            exp_buffer.append([obs, action, next_obs, reward, done]) #GT
            
            # After certain steps, we train our Q network with samples from the experience replay buffer
            if global_step % steps_train == 0 and global_step > start_steps:
                #print('i:', i, 'Learn: ', global_step, start_steps, steps_train)
                
                # sample experience
                o_obs, o_act, o_next_obs, o_rew, o_done = sample_memories(batch_size)

                # states
                o_obs = [x for x in o_obs]

                # next states
                o_next_obs = [x for x in o_next_obs]

                # next actions
                if filename.startswith('original'):
                    next_act = mainQ_outputs.eval(feed_dict={X:o_next_obs, in_training_mode:False})                
                else:
                    next_act = targetQ_outputs.eval(feed_dict={X:o_next_obs, in_training_mode:False})  #GT
    
                # reward
                y_batch = o_rew + discount_factor * np.max(next_act, axis=-1) * (1-o_done) 

                # merge all summaries and write to the file
                mrg_summary = merge_summary.eval(feed_dict={X:o_obs, y:np.expand_dims(y_batch, axis=-1), X_action:o_act, in_training_mode:False})
                file_writer.add_summary(mrg_summary, global_step)

                # now we train the network and calculate loss
                train_loss, _ = sess.run([loss, training_op], feed_dict={X:o_obs, y:np.expand_dims(y_batch, axis=-1), X_action:o_act, in_training_mode:True})
                episodic_loss.append(train_loss)
            
            # after some interval we copy our main Q network weights to target Q network
            if (global_step+1) % copy_steps == 0 and global_step > start_steps:
                #print('i:', i, 'Replace: ', global_step+1, start_steps, copy_steps)
                copy_target_to_main.run()
                
            obs = next_obs
            epoch += 1
            global_step += 1
            episodic_reward += reward
        
        all_episodic_reward.append(episodic_reward)
        print('i:', i, 'Epoch:', epoch, 'Reward:', episodic_reward)
        
        plt.plot(all_episodic_reward)
        plt.ylabel('Rewards')
        plt.xlabel('training steps')
        plt.title(filename, fontsize=20)
        plt.savefig(logdir+os.sep+filename+".png", bbox_inches='tight', dpi=100)
        plt.close()
        
        with open(logdir+os.sep+filename+'.csv', 'a') as f: # append to the file created
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow([i+1, episodic_reward])


# In[ ]:




