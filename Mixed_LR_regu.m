clear all;

% set the random seed
rng(42);

% number of failing instances for EM algorithm with different
% initialization
fail_count_EM = 0;
fail_count_EMus = 0;
fail_count_EMcs = 0;
% the gap between the final result and the optimal solution
gap_EM = 0;
gap_EMus = 0;
gap_EMcs = 0;
% number of total iterations
N_iter_EM = 0;
N_iter_EMus = 0;
N_iter_EMcs = 0;

N_trial = 1000;
d = 8; % dimension
N = 1000; % number of data points
K = 8; % clusters
mu = 0.01; % regularization term parameter
noise_level = 1e-2;
iter_max = 10000; 
N_sample = 1; % # of centers sampled each time in careful seeding

for n = 1:N_trial
    % generate the true solution
    x_star = zeros(d,K); 
    x_star(:,1) = randn(d,1); % true solution
    for k = 2:K
        % x_star(:,k) = x_star(:,1) + 0.3 * randn(d,1);
        x_star(:,k) = randn(d,1);
    end
    % the objective function is 1/2 ||ax-b||^2 + mu/2 ||x||^2
    a = randn(N,d); % visible data points
    mask = randi([1,K],N,1); % determine the cluster
    b = zeros(N,1); % visible data points
    for k = 1:K
        if sum(mask == k) > 0
            b(mask == k) = a(mask == k,:) * x_star(:,k) + noise_level * randn(sum(mask == k),1);
        end
    end
    
    x0 = randn(d,K);  % Guassian initialization
    [x_EM, loss_EM] = EM(a, b, mu, x0, iter_max);

    x0_us = uniform_seeding(a, b, mu, K); % uniform index initialization
    [x_EMus, loss_EMus] = EM(a, b, mu, x0_us, iter_max);


    x0_cs = careful_seeding(a, b, mu, K, N_sample);
    [x_EMcs, loss_EMcs] = EM(a, b, mu, x0_cs, iter_max);
    
    loss_xstar = Loss(a, b, mu, x_star);
    
    % number of total iterations for the initialization methods
    N_iter_EM = N_iter_EM + length(loss_EM) - 1;
    N_iter_EMus = N_iter_EMus + length(loss_EMus) - 1;
    N_iter_EMcs = N_iter_EMcs + length(loss_EMcs) - 1;

    if loss_EM(length(loss_EM)) > loss_xstar
        fail_count_EM = fail_count_EM + 1;
        gap_EM = gap_EM + loss_EM(length(loss_EM)) - loss_xstar;
    end
    
    if loss_EMus(length(loss_EMus)) > loss_xstar
        fail_count_EMus = fail_count_EMus + 1;
        gap_EMus = gap_EMus + loss_EMus(length(loss_EMus)) - loss_xstar;
    end

    if loss_EMcs(length(loss_EMcs)) > loss_xstar
        fail_count_EMcs = fail_count_EMcs + 1;
        gap_EMcs = gap_EMcs + loss_EMcs(length(loss_EMcs)) - loss_xstar;
    end
end

fprintf('failure probability of EM: %f\n', fail_count_EM / N_trial);
fprintf('averaged iteration number of EM: %f\n', N_iter_EM / N_trial);
fprintf("failure probability of EM with uniform seeding: %f\n", fail_count_EMus / N_trial);
fprintf('averaged iteration number of EM with uniform seeding: %f\n', N_iter_EMus / N_trial);
fprintf("failure probability of EM with careful seeding: %f\n", fail_count_EMcs / N_trial);
fprintf('averaged iteration number of EM with careful seeding: %f\n', N_iter_EMcs / N_trial);

% semilogy(1:length(loss_EM), loss_EM, '*-');
% hold on;
% semilogy(1:length(loss_EMcs), loss_EMcs, '*-');
% hold on;
% 
% L = max(length(loss_EM), length(loss_EMcs));
% plot(1:L, loss_xstar * ones(L,1), '*-');
% legend('EM algorithm with random initialization', 'EM algorithm with careful seeding', 'loss at x*');

%% compute the loss function
function loss = Loss(a, b, mu, x)
    N = length(b);
    [~, K] = size(x);
    loss_temp = zeros(N, K);
    for k = 1:K
        loss_temp(:,k) = 1/2 * (a * x(:,k) - b) .^2 + 1/2 * mu * norm(x(:,k)) ^ 2;
    end
    loss = sum(min(loss_temp, [], 2)) / N;
end

%% initialization using random index uniform sampling
function x = uniform_seeding(a, b, mu, K)
    [N, d] = size(a);
    x = zeros(d, K);

    I_sampled = randperm(N, K);
    for k = 1:K
        x(:,k) = (a(I_sampled(k),:)' * a(I_sampled(k),:) + mu * eye(d)) \ (a(I_sampled(k),:)' * b(I_sampled(k)));
    end

end

%% initialization inspired by Arthur, D., & Vassilvitskii, S. (2007, January). K-means++ the advantages of careful seeding. In Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms (pp. 1027-1035).
function x = careful_seeding(a, b, mu, K, N_sample)
    [N, d] = size(a);
    x = zeros(d, K);

    I_sampled = randi([1,N], N_sample, 1);  % uniformly sample the first token
    x(:,1) = (a(I_sampled,:)' * a(I_sampled,:) + mu * eye(d)) \ (a(I_sampled,:)' * b(I_sampled)); 
    grad_square_min = sum(((a * x(:,1) - b) .* a + mu * ones(N,1) * x(:,1)') .^2, 2);

    for k = 2:K
        weights = grad_square_min / sum(grad_square_min);
        I_sampled = randsample(N, N_sample, true, weights);
        x(:,k) = (a(I_sampled,:)' * a(I_sampled,:) + mu * eye(d)) \ (a(I_sampled,:)' * b(I_sampled));
        grad_square_min = min(grad_square_min, sum(((a * x(:,k) - b) .* a + mu * ones(N,1) * x(:,k)') .^2, 2));
    end
end

%% EM algorithm
function [x, loss] = EM(a, b, mu, x0, iter_max)
    x = x0;
    loss = zeros(iter_max+1,1);
    N = length(b);
    [d, K] = size(x);
    loss_temp = zeros(N, K);
    min_loc_old = zeros(N,1);

    iter = 1;
    while true
        for k = 1:K
            loss_temp(:,k) = 1/2 * (a * x(:,k) - b) .^2 + 1/2 * mu * norm(x(:,k)) ^ 2;
        end
        [min_value, min_loc] = min(loss_temp, [], 2);
        loss(iter) = sum(min_value) / N;
        if sum(abs(min_loc-min_loc_old)) == 0 || iter >= iter_max + 1
            break
        end
        
        min_loc_old = min_loc;
        for k = 1:K
            if ~isempty(min_loc(min_loc == k))
                x(:,k) = (a(min_loc==k,:)' * a(min_loc==k,:) + sum(min_loc == k) * ...
                    mu * eye(d)) \ (a(min_loc==k,:)' * b(min_loc==k));
            end
        end
        iter = iter + 1;
    end

    loss = loss(1:iter);
end