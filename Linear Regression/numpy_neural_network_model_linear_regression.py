# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# we are about to create a two variable linear model f(x,z) = ax + bz + c
observations = 1000

# uniform - generate random values from uniform distribution, each values have equal chance to generate
xs = np.random.uniform(low=-10, high=10, size=(observations, 1)) # [1000, 1] size = (n x k), n - numbers of values, k - numbers of variables 
zs = np.random.uniform(-10, 10, (observations,1)) # [1000, 1]

# n x k = 1000 x 2 
# np.column_stack(appropriate tuples) takes a sequence of 1D arrays and stacks them into a single 2D array
inputs = np.column_stack((xs,zs)) # shape [1000, 2]

# in supervised learning we must know two major parameters: inputs, targets | others: weights, biases, outputs - compute computer
# create the targets we will aim at: targets = f(x,z) = 2x - 3z + 5 + noise (correct results), noise its introduced to randomize our data
noise = np.random.uniform(-1, 1, (observations,1))
targets = 2*xs - 3*zs + 5 + noise # shape [1000, 1]

# %%
# plot the training data - the point is to see that there is a strong trend that our model should learn to reproduce
targets = targets.reshape(observations,) # [1000,1] - array(column) -> [1000,] - list | with 2d to 1d because we want to visualize
fig = plt.figure() # declare the figure
ax = fig.add_subplot(111, projection='3d') # method allows to create 3d plot
ax.plot(xs, zs, targets) # choose the axes: x, y, z
ax.set_xlabel('xs') # set labels
ax.set_ylabel('zs')
ax.set_zlabel('targets')
ax.view_init(azim=100) # plot the data from different angles. azim [0;200] - can change
plt.show() # unhidden if you want
targets = targets.reshape(observations,1)

# %%
# we don't want to start from any arbitrary number. Rather, we randomly select some small initial weights
init_range = 0.1 # our initial weights and biases will be picked randomly from [-0.1, 0.1]
weights = np.random.uniform(-init_range, init_range, size=(2,1)) # size 2,1 because we have to 2 variables, so we have 2 weights and single output
biases = np.random.uniform(-init_range, init_range, size=1) # scalar - in machine learning, there are as many biases, as there are outputs
print(f"For our model weights are: \n {weights}")
print(f"For our model biases is: \n {biases}")

# set a learning rate
learning_rate = 0.02

# %%
# train the model
# because it's regression we have to use l2-norm loss/2 (gradient descent algorithm iteration)
# game plan for each iteration: 1. Calculate outputs, 2. Compare outputs to targets through the loss, 3. Print the loss, 4. Adjust weights and biases -> repeated/iterations
for i in range(100):
    outputs = np.dot(inputs, weights) + biases # np.dot - method used for multiplying matrices (or A.dot(B) = equal np.dot) [1000,1] x [1,2] -> [1000,1]
    # python adds scalars to matrices element-wise. (one by one)
    deltas = outputs - targets
    # l2-norm loss formula = Sum(yi - ti)^2, # division by a constant (2 or observations) doesn't change the logic of the loss function, doesn't change direction optimization, only scale of optimization - constant doesn't change derivatives using in algorithm
    loss = np.sum(deltas ** 2)  / 2 / observations # we divided by observations = mean (average) loss by observation
    print(loss)
    deltas_scaled = deltas / observations
    weights = weights - learning_rate * np.dot(inputs.T, deltas_scaled) # [2,1] = [2,1] - [1] * [1000,2] - we must transpose -> [2,1000] we can multiply thanks transpose, [1000,1]
    biases = biases - learning_rate * np.sum(deltas_scaled)
    # in the memory of the computer, the variables: weights, biases and outputs contain their optimized values: those from the last iteration of the loop

# %%
# to fit our algorithm to solution we have to manipulate learning rate, number of observations, number of iterations and initial range for initializing weights and biases #f(x,z) = 2x - 3z + 5 + noise
print(weights, biases)
# %%
# checking: 45 degree -> good fit - same line y = x 
plt.plot(outputs, targets)
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()

# %%
