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

# save the results into a txt file
def write_to_file(filename, text):
    with open(filename, "a") as file:
        file.write(text + "\n")

# Get the current date and time
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

# Create a filename with the date and time
filename = f"result_{formatted_time}.txt"

# number of failing instances for EM, EMus and EMcs algorithms
fail_count_EM = 0
fail_count_EM_us = 0
fail_count_EM_cs = 0

N_iter_EM = 0
N_iter_EM_us = 0
N_iter_EM_cs = 0

train_loss_EM = 0
train_loss_EM_us = 0
train_loss_EM_cs = 0

test_loss_EM = 0
test_loss_EM_us = 0
test_loss_EM_cs = 0

# hyperparameters for the experiment
N_trial = 100
d = 10  # Input dimension
hidden_dim = 5   # Hidden layer dimension
K = 5 # clusters
N = 1000 # number of data points
mu = 0.01  # regularization paramter
noise_level = 1e-2
inner_iter = 10  # inner iteration number seeking for an approximate class minima
iter_max = 15
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

write_to_file(filename, hyperparameters)

# If CUDA is used
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups


for n in range(N_trial):
    print("start trial ", n)
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

    net_init = [TwoLayerNet(d, hidden_dim).to(device) for i in range(K)]  # randomly initialize the nets
    print("EM algorithm for random Gaussian initialization")
    nets_EM, loss_EM, iter_EM = EM_train(x, y, mu, net_init, inner_iter, iter_max, loss_net_star, device)  # nets_EM is the nets trained

    net_init_us = uniform_seeding(x, y, mu, K, d, hidden_dim, epochs=seeding_epochs)
    print("EM algorithm for uniform seeding initialization")
    nets_EM_us, loss_EM_us, iter_EM_us = EM_train(x, y, mu, net_init_us, inner_iter, iter_max, loss_net_star, device)

    net_init_cs = careful_seeding(x, y, mu, K, d, hidden_dim, epochs=seeding_epochs)
    print("EM algorithm for careful seeding initialization")
    nets_EM_cs, loss_EM_cs, iter_EM_cs = EM_train(x, y, mu, net_init_cs, inner_iter, iter_max, loss_net_star, device)

    # compute the loss of net_star
    # the loss function for each (x,y) pair is 1/2 * ||f(x,theta)-y||^2 + 1/2 * mu * ||theta||^2

    # cumulate the N_iter
    N_iter_EM = N_iter_EM + iter_EM
    N_iter_EM_us = N_iter_EM_us + iter_EM_us
    N_iter_EM_cs = N_iter_EM_cs + iter_EM_cs

    train_loss_EM = train_loss_EM + Loss(nets_EM, x, y, 0.)
    train_loss_EM_us = train_loss_EM_us + Loss(nets_EM_us, x, y, 0.)
    train_loss_EM_cs = train_loss_EM_cs + Loss(nets_EM_cs, x, y, 0.)

    test_loss_EM = test_loss_EM + Loss(nets_EM, x_test, y_test, 0.)
    test_loss_EM_us = test_loss_EM_us + Loss(nets_EM_us, x_test, y_test, 0.)
    test_loss_EM_cs = test_loss_EM_cs + Loss(nets_EM_cs, x_test, y_test, 0.)

    if iter_EM == iter_max:
        fail_count_EM = fail_count_EM + 1

    if iter_EM_us == iter_max:
        fail_count_EM_us = fail_count_EM_us + 1

    if iter_EM_cs == iter_max:
        fail_count_EM_cs = fail_count_EM_cs + 1

    # Writing results to the file
    result_text = f"Trial {n}, Loss_EM: {loss_EM}, Loss_EM_us: {loss_EM_us}, " \
                  f"Loss_EM_cs: {loss_EM_cs}, Loss_net_star: {loss_net_star}; " \
                  f"Failing rate, EM: {fail_count_EM/(n+1)}, EM_us: {fail_count_EM_us/(n+1)}, " \
                  f"EM_cs: {fail_count_EM_cs/(n+1)}; Average iterations, EM: {N_iter_EM/(n+1)}, " \
                  f"EM_cs: {N_iter_EM_cs/(n+1)}, EM_us: {N_iter_EM_us/(n+1)}"
    print(result_text)
    write_to_file(filename, result_text)



# Write final statistics
final_stats = f"""
Failure probability of EM: {fail_count_EM / N_trial}
Average iterations of EM: {N_iter_EM/N_trial}
Training loss of EM: {train_loss_EM/N_trial}
Testing loss of EM: {test_loss_EM/N_trial}
Failure probability of EM with uniform seeding: {fail_count_EM_us / N_trial}
Average iterations of EM_us: {N_iter_EM_us/N_trial}
Training loss of EM_us: {train_loss_EM_us/N_trial}
Testing loss of EM_us: {test_loss_EM_us/N_trial}
Failure probability of EM with careful seeding: {fail_count_EM_cs / N_trial}
Average iterations of EM_cs: {N_iter_EM_cs/N_trial}
Training loss of EM_cs: {train_loss_EM_cs/N_trial}
Testing loss of EM_cs: {test_loss_EM_cs/N_trial}
"""
print(final_stats)

write_to_file(filename, "Final Statistics:")
write_to_file(filename, final_stats)