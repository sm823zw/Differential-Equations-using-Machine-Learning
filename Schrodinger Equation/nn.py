import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd
from autograd.misc.optimizers import adam

# Initialize neural network weights
def initialize_nn(scale, n_neurons_per_layer):
    params = []
    for i, j in zip(n_neurons_per_layer[:-1], n_neurons_per_layer[1:]):
        params.append((np.random.randn(i, j)*scale, np.random.randn(j)*scale))
    return params

# Feed-forward the inputs
def psi(nn_params, inputs):
    for W, b in nn_params:
        outputs = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)
    return outputs

# Ensure derivatives and double derivatives will be computed
psi_x = autograd.elementwise_grad(psi, 1)
psi_xx = autograd.elementwise_grad(psi_x, 1)

# Create the neural network with given architecture
n_neurons_per_layer=[1, 40, 1]
nn_params = initialize_nn(0.1, n_neurons_per_layer)
params = {'nn_params': nn_params, 'E': 0.5}


L = 1
x = np.linspace(0, L, 50)[:, None]

# Define loss function
def loss_function(params, step):
    nn_params = params['nn_params']
    E = params['E']

    first_term = - psi_xx(nn_params, x)  - E * psi(nn_params, x) 

    bc0 = psi(nn_params, 0.0)
    bc1 = psi(nn_params, L)

    psi_sq = psi(nn_params, x)**2

    prob = np.sum((psi_sq[1:] + psi_sq[0:-1]) / 2 * (x[1:] - x[0:-1]))
    
    loss = np.mean(first_term**2) + bc0**2 + bc1**2 + (1.0 - prob)**2
    
    return loss

# Define callback to observe training loss
def callback(params, step, g):
    if step % 100 == 0:
        print("Epoch : {0:3d}, loss : {1}".format(step, loss_function(params, step)[0][0]))


# Train the network and update weights using adam optimizer
params = adam(autograd.grad(loss_function), params, step_size=0.01, num_iters=5001, callback=callback) 

# Print the eigenvalue obtained
print(params['E'])

# Plot the results for 1000 test samples
N = 1000
x = np.linspace(0, L, N)[:, None]
y = psi(params['nn_params'], x)
error = 1/N*np.sum(y**2 - (np.sqrt(2/L)*np.sin(np.pi* x/L))**2)**2
plt.plot(x, y**2, label='NN')
plt.plot(x, (np.sqrt(2/L)*np.sin(np.pi* x/L))**2, 'r--', label='analytical')
plt.ylabel('$\psi(x)^2$')
plt.xlabel('x')
plt.title('$L=1$, $N=50$, error =' + str(error))
plt.legend()
plt.show()
