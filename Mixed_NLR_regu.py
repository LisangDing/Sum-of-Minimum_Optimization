import torch
import torch.nn as nn
import torch.nn.functional as F



class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TwoLayerNet, self).__init__()
        # First layer (input to hidden)
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        # Second layer (hidden to output)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Pass the input through the hidden layer, then apply ReLU activation
        x = F.relu(self.hidden_layer(x))
        # Pass the output of the hidden layer to the output layer
        x = self.output_layer(x)
        return x



def EM_train(x, y, mu, nets_init, inner_iter, iter_max):
    return None, None

def uniform_seeding(x, y, mu, K):
    return None

def careful_seeding(x, y, mu, K, N_sample):
    return None

def Loss(nets, x, y, mu):
    # nets contained the output of the EM algorithm
    losses = []
    for net in nets:
        reg_loss = 0.5 * mu * (torch.sum(net.hidden_layer.weight ** 2) + torch.sum(net.output_layer.weight ** 2))
        losses.append(0.5 * (net(x) - y) ** 2 + reg_loss)

    stacked_losses = torch.stack(losses) # stack the tensors to a (K, N) tensor
    min_values, _ = torch.min(stacked_losses, dim=0)
    result = torch.sum(min_values)
    return result


# number of failing instances for EM, EMus and EMcs algorithms
fail_count_EM = 0
fail_count_EM_us = 0
fail_count_EM_cs = 0
# number of total iterations
N_iter_EM = 0
N_iter_EMus = 0
N_iter_EMcs = 0

# hyperparameters for the experiment
N_trial = 1000
d = 10  # Input dimension
hidden_dim = 5   # Hidden layer dimension
K = 5 # clusters
N = 1000 # number of data points
mu = 0.01  # regularization paramter
noise_level = 1e-2
inner_iter = 10  # inner iteration number seeking for an approximate class minima
iter_max = 10000
N_sample = 1 # number of centers sampled each time in careful seeding


# Set the seed for generating random numbers
seed_value = 42
torch.manual_seed(seed_value)

# If CUDA is used
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups


for n in range(N_trial):
    # generate the K true solutions
    nets_star = [TwoLayerNet(d, hidden_dim) for i in range(K)]

    # generate the data points
    x = torch.randn(N, d) # inputs
    x_class = torch.randint(0, K, (N,))  # determine the input of x
    y = torch.zeros(N) # outputs

    for i in range(N):
        y[i] = nets_star[x_class[i]](x[i,:])

    # add noise to y
    noise = noise_level * torch.randn(N,)
    y = y + noise


    nets_init = [TwoLayerNet(d, hidden_dim) for i in range(K)]  # randomly initialize the nets
    nets_EM, loss_EM = EM_train(x, y, mu, nets_init, inner_iter, iter_max)  # nets_EM is the nets trained

    nets_init_us = uniform_seeding(x, y, mu, K)
    nets_EM_us, loss_EM_us = EM_train(x, y, mu, nets_init_us, inner_iter, iter_max)

    nets_init_cs = careful_seeding(x, y, mu, K, N_sample)
    nets_EM_cs, loss_EM_cs = EM_train(x, y, mu, nets_init_cs, inner_iter, iter_max)

    # compute the loss of nets_star
    # the loss function for each (x,y) pair is 1/2 * ||f(x,theta)-y||^2 + 1/2 * mu * ||theta||^2

    loss_nets_star = Loss(nets_star, x, y, mu)

    N_iter_EM = N_iter_EM + len(loss_EM) - 1
    N_iter_EMus = N_iter_EMus + len(loss_EM_us) - 1
    N_iter_EMcs = N_iter_EMcs + len(loss_EM_cs) - 1

    if loss_EM[-1] > loss_nets_star:
        fail_count_EM = fail_count_EM + 1

    if loss_EM_us[-1] > loss_nets_star:
        fail_count_EM_us = fail_count_EM_us + 1

    if loss_EM_cs[-1] > loss_nets_star:
        fail_count_EM_cs = fail_count_EM_cs + 1


    # Give statisticts on the algorithms