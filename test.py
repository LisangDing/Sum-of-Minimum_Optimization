import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import datetime


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



def EM_train(x, y, mu, net_init, inner_iter, iter_max, loss_net_star, device, learning_rate=1e-3):
    N = x.shape[0]  # Number of data points
    K = len(net_init)  # Number of networks

    # Optimizer list corresponding to each network
    optimizers = [torch.optim.Adam(net.parameters(), lr=learning_rate) for net in net_init]

    for iter in range(iter_max):
        if iter % 10 == 0:
            print("EM iteration ", iter)
        # Expectation Step: Assign data points to the closest network
        # classes = [[] for _ in range(K)]
        # for i in range(N):
        #     losses = []
        #     for j in range(K):
        #         # Calculate loss for network j and data point i
        #         y_pred = net_init[j](x[i, :])
        #         loss = 0.5 * (y_pred - y[i]) ** 2 + 0.5 * mu * sum(
        #             torch.norm(p, p=2) ** 2 for p in net_init[j].parameters())
        #         losses.append(loss.item())
        #
        #     # Assign data point to the class with minimum loss
        #     min_loss_class = losses.index(min(losses))
        #     classes[min_loss_class].append(i)

        # Initialize a K x N tensor for storing losses
        losses = torch.zeros(K, N, device=device)

        for j in range(K):
            # Compute predictions for all data points with network j
            y_pred = net_init[j](x).squeeze()

            # Compute prediction loss for all data points
            pred_loss = 0.5 * (y_pred - y) ** 2

            # Compute regularization loss for network j
            reg_loss = 0.5 * mu * sum(torch.norm(p, p=2) ** 2 for p in net_init[j].parameters())

            # Add prediction loss and regularization term
            losses[j, :] = pred_loss + reg_loss

        # Assign data point to the class with minimum loss
        min_loss, min_loss_classes = torch.min(losses, dim=0)

        # compare the current loss and the loss of net_star
        current_loss = torch.sum(min_loss) / N

        # stop training if the current loss is already smaller than the loss of net_star
        if current_loss < loss_net_star:
            return net_init, current_loss, iter

        # Initialize classes
        classes = [[] for _ in range(K)]

        # Populate the classes with data point indices
        for i in range(N):
            classes[min_loss_classes[i]].append(i)

        # Maximization Step: Train each network on its assigned class
        for j in range(K):
            if classes[j]:  # Check if class is not empty
                # Gather all x[i, :] and y[i] for i in classes[j]
                X = torch.stack([x[i, :] for i in classes[j]])
                Y = torch.stack([y[i] for i in classes[j]])
                for _ in range(inner_iter):
                    # Forward pass for the batch
                    Y_pred = net_init[j](X).squeeze()
                    # Compute the loss
                    loss = 0.5 * torch.norm(Y_pred - Y) ** 2
                    reg_loss = 0.5 * mu * sum(torch.norm(p, p=2) ** 2 for p in net_init[j].parameters())


                    average_loss = loss / len(classes[j]) + reg_loss

                    # Zero gradients, perform a backward pass, and update the weights.
                    optimizers[j].zero_grad()
                    average_loss.backward()
                    optimizers[j].step()



        # Compute final EM loss
    EM_loss = sum([min([0.5 * (net_init[j](x[i, :]) - y[i]) ** 2 + 0.5 * mu * sum(
        torch.norm(p, p=2) ** 2 for p in net_init[j].parameters()) for j in range(K)]) for i in range(N)]) / N

    return net_init, EM_loss, iter_max

def uniform_seeding(x, y, mu, K, d, hidden_dim, learning_rate=1e-3, epochs=300):
    N = x.shape[0]  # Number of data points
    # Initialize K two-layer networks
    net_init = [TwoLayerNet(d, hidden_dim).to(device) for _ in range(K)]

    # Sample K indices from range N
    I_sampled = random.sample(range(N), K)

    # Optimizer list corresponding to each network
    optimizers = [torch.optim.Adam(net.parameters(), lr=learning_rate) for net in net_init]

    for k in range(K):
        i = I_sampled[k]
        x_sample = x[i, :]
        y_sample = y[i]

        for epoch in range(epochs):
            # Forward pass
            y_pred = net_init[k](x_sample)
            # Compute the loss
            loss = 0.5 * (y_pred - y_sample) ** 2 +\
                   0.5 * mu * sum(torch.norm(p, p=2) ** 2 for p in net_init[k].parameters())

            # Zero gradients, perform a backward pass, and update the weights.
            optimizers[k].zero_grad()
            loss.backward()
            optimizers[k].step()
    return net_init

def careful_seeding(x, y, mu, K, d, hidden_dim, learning_rate=1e-3, epochs=300):
    N = x.shape[0]  # Number of data points
    # Initialize K two-layer networks
    net_init = [TwoLayerNet(d, hidden_dim).to(device) for _ in range(K)]

    # Optimizer list corresponding to each network
    optimizers = [torch.optim.Adam(net.parameters(), lr=learning_rate) for net in net_init]

    # Sample one index from range N
    I_sampled = random.sample(range(N), 1)[0]
    sampled_set = [I_sampled]

    x_sample = x[I_sampled, :]
    y_sample = y[I_sampled]

    # train net[0] using the uniform sampled (x,y) pair
    for epoch in range(epochs):
        # Forward pass
        y_pred = net_init[0](x_sample)

        # Compute the loss
        loss = 0.5 * (y_pred - y_sample) ** 2 + \
               0.5 * mu * sum(torch.norm(p, p=2) ** 2 for p in net_init[0].parameters())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizers[0].zero_grad()
        loss.backward()
        optimizers[0].step()


    for k in range(1,K):
        grad_square = [math.inf] * N
        # Calculate grad_square for each data point
        for i in range(N):
            if i in sampled_set:
                grad_square[i] = 0
            else:
                for j in range(k):
                    # Zero gradients
                    optimizers[j].zero_grad()

                    # Forward pass
                    y_pred = net_init[j](x[i, :])

                    # Compute the loss
                    loss = 0.5 * (y_pred - y[i]) ** 2 + \
                           0.5 * mu * sum(torch.norm(p, p=2) ** 2 for p in net_init[j].parameters())

                    # Backward pass to get gradient
                    loss.backward()

                    # Compute the squared gradient
                    grad_square[i] = min(grad_square[i],
                                         sum(p.grad.norm() ** 2 for p in net_init[j].parameters()
                                             if p.grad is not None))

        # Normalize grad_square
        total = sum(grad_square)
        probabilities = [g / total for g in grad_square]

        # Sample an index based on grad_square
        I_sampled = random.choices(range(N), weights=probabilities, k=1)[0]
        sampled_set = sampled_set + [I_sampled]

        x_sample = x[I_sampled, :]
        y_sample = y[I_sampled]

        # Train net[k] using the sampled (x, y) pair
        for epoch in range(epochs):
            # Forward pass
            y_pred = net_init[k](x_sample)

            # Compute the loss
            loss = 0.5 * (y_pred - y_sample) ** 2 + \
                   0.5 * mu * sum(torch.norm(p, p=2) ** 2 for p in net_init[k].parameters())

            # Zero gradients, perform a backward pass, and update the weights.
            optimizers[k].zero_grad()
            loss.backward()
            optimizers[k].step()

    return net_init


def Loss(nets, x, y, mu):
    N = x.shape[0]  # Number of data points
    K = len(nets)  # Number of networks

    # Initialize a tensor to store losses for each network and data point
    losses = torch.zeros(K, N)

    for j, net in enumerate(nets):
        # Forward pass for all data points
        y_pred = net(x).squeeze()

        # Calculate prediction loss for each data point
        pred_loss = 0.5 * (y_pred - y) ** 2

        # Regularization term: sum of squares of weights
        reg_loss = 0.5 * mu * sum(torch.norm(p, p=2) ** 2 for p in net.parameters())

        # Add prediction loss and regularization term
        losses[j,:] = pred_loss + reg_loss

    # Find minimum loss for each data point across all networks
    min_values, _ = torch.min(losses, dim=0)

    # Sum up the minimum losses for all data points
    result = torch.sum(min_values) / N
    return result




# hyperparameters for the experiment
N_trial = 100
d = 5  # Input dimension
hidden_dim = 3   # Hidden layer dimension
K = 5 # clusters
N = 1000 # number of data points
mu = 0.01  # regularization paramter
noise_level = 1e-2
inner_iter = 10  # inner iteration number seeking for an approximate class minima
iter_max = 300
N_sample = 1 # number of centers sampled each time in careful seeding
seeding_epochs = 300
N_test = 200


# Set the seed for generating random numbers
seed_value = 42
torch.manual_seed(seed_value)


# GPU settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Save hyperparameters
hyperparameters = f"""
N_trial: {N_trial}
d: {d}
hidden_dim: {hidden_dim}
K: {K}
N: {N}
mu: {mu}
noise_level: {noise_level}
inner_iter: {inner_iter}
iter_max: {iter_max}
N_sample: {N_sample}
seeding_epochs: {seeding_epochs}
N_test: {N_test}
"""

print(hyperparameters)


# If CUDA is used
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups

gt_loss = 0


for n in range(N_trial):
    # generate the K true solutions
    net_star = [TwoLayerNet(d, hidden_dim).to(device) for _ in range(K)]
    
    # Fix the parameters in net_star
    for net in net_star:
        for param in net.parameters():
            param.requires_grad = False

    # generate the data points
    x = torch.randn(N, d) # inputs
    x_class = torch.randint(0, K, (N,)).to(device)  # determine the input of x
    y = torch.zeros(N) # outputs

    # move the data to device
    x = x.to(device)
    y = y.to(device)

    for i in range(N):
        y[i] = net_star[x_class[i]](x[i,:])

    # add noise to y
    noise = noise_level * torch.randn(N,).to(device)
    y = y + noise

    # generate the testing data
    x_test = torch.randn(N_test, d)  # inputs
    x_test_class = torch.randint(0, K, (N_test,)).to(device)  # determine the input of x
    y_test = torch.zeros(N_test)  # outputs

    # move the data to device
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    for i in range(N_test):
        y_test[i] = net_star[x_test_class[i]](x_test[i, :])

    # add noise to y
    noise_test = noise_level * torch.randn(N_test, ).to(device)
    y_test = y_test + noise_test


    # loss on the ground truth networks
    loss_net_star = Loss(net_star, x, y, mu)

    gt_loss = gt_loss + loss_net_star





print(gt_loss / N_trial)