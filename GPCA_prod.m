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
% matching loss
match_loss_EM = 0;
match_loss_EMus = 0;
match_loss_EMcs = 0;

N_trial = 1000;
d = 4; % dimension
r = d - 2; % PCA complementary space
N = 1000; % number of data points
K = 3; % clusters
noise_level = 1e-2;
iter_max = 50; 

for n = 1:N_trial
    v_star = zeros(K,d,2); % ground truth principle components
    for k = 1:K
        % Generate two random vectors
        v1 = randn(d,1);
        v2 = randn(d,1);

        % Normalize v1
        v1 = v1 / norm(v1);
        % Make v2 orthogonal to v1
        v2 = v2 - dot(v2, v1) * v1;
        % Normalize v2
        v2 = v2 / norm(v2);
        v_star(k,:,:) = [v1, v2];
    end

    a = randn(N,2); % factors of two PC
    mask = randi([1,K],N,1); % determine the cluster

    x = zeros(N,d);
    for i = 1:N
        x(i,:) = v_star(mask(i),:,1) * a(i,1) * 1 + ...
            v_star(mask(i),:,2) * a(i,2) * 0.2;
    end

    % Generate K copies of d * r initializations
    A0 = normal_init(K, d, r);
    [A_EM, loss_EM] = BCD(x, A0, iter_max);

    A0_us = uniform_seeding(x, K, r); % uniform index initialization
    [A_EMus, loss_EMus] = BCD(x, A0_us, iter_max);

    A0_cs = careful_seeding(x, K, r);
    [A_EMcs, loss_EMcs] = BCD(x, A0_cs, iter_max);
    
    loss_Astar = sum(Loss(x, v_star));
    
    % number of total iterations for the initialization methods
    N_iter_EM = N_iter_EM + length(loss_EM) - 1;
    N_iter_EMus = N_iter_EMus + length(loss_EMus) - 1;
    N_iter_EMcs = N_iter_EMcs + length(loss_EMcs) - 1;

    % compute the matching loss
    match_loss_EM = match_loss_EM + Loss_A_matching(x, A_EM, mask);
    match_loss_EMus = match_loss_EMus + Loss_A_matching(x, A_EMus, mask);
    match_loss_EMcs = match_loss_EMcs + Loss_A_matching(x, A_EMcs, mask);

    if loss_EM(length(loss_EM)) > loss_Astar
        fail_count_EM = fail_count_EM + 1;
        gap_EM = gap_EM + loss_EM(length(loss_EM)) - loss_Astar;
    end
    
    if loss_EMus(length(loss_EMus)) > loss_Astar
        fail_count_EMus = fail_count_EMus + 1;
        gap_EMus = gap_EMus + loss_EMus(length(loss_EMus)) - loss_Astar;
    end

    if loss_EMcs(length(loss_EMcs)) > loss_Astar
        fail_count_EMcs = fail_count_EMcs + 1;
        gap_EMcs = gap_EMcs + loss_EMcs(length(loss_EMcs)) - loss_Astar;
    end
end

fprintf('K=%d, d=%d \n', K, d);
fprintf('failure probability of EM: %f\n', fail_count_EM / N_trial);
fprintf('averaged iteration number of EM: %f\n', N_iter_EM / N_trial);
fprintf('averaged gap of EM: %f\n', gap_EM / N_trial);
fprintf('averaged matching loss of EM: %f\n', match_loss_EM / N_trial);
fprintf("failure probability of EM with uniform seeding: %f\n", fail_count_EMus / N_trial);
fprintf('averaged iteration number of EM with uniform seeding: %f\n', N_iter_EMus / N_trial);
fprintf('averaged gap of EM_us: %f\n', gap_EMus / N_trial);
fprintf('averaged matching loss of EM_us: %f\n', match_loss_EMus / N_trial);
fprintf("failure probability of EM with careful seeding: %f\n", fail_count_EMcs / N_trial);
fprintf('averaged iteration number of EM with careful seeding: %f\n', N_iter_EMcs / N_trial);
fprintf('averaged gap of EM_cs: %f\n', gap_EMcs / N_trial);
fprintf('averaged matching loss of EM_cs: %f\n', match_loss_EMcs / N_trial);

% semilogy(1:length(loss_EM), loss_EM, '*-');
% hold on;
% semilogy(1:length(loss_EMcs), loss_EMcs, '*-');
% hold on;
% 
% L = max(length(loss_EM), length(loss_EMcs));
% plot(1:L, loss_xstar * ones(L,1), '*-');
% legend('EM algorithm with random initialization', 'EM algorithm with careful seeding', 'loss at x*');

%% least matching distance
function distance = minDistance(a, b, K)
    % All possible permutations of 1:K
    permsK = perms(1:K);

    % Initialize minimum distance to the maximum possible value
    minDist = length(a);

    % Iterate over all permutations
    for i = 1:size(permsK, 1)
        % Apply the permutation to a
        a_permute = arrayfun(@(x) permsK(i, x), a);

        % Calculate the number of mismatches
        mismatches = sum(a_permute ~= b);

        % Update minimum distance
        if mismatches < minDist
            minDist = mismatches;
        end
    end

    % Return the minimum distance found
    distance = minDist;
end


%% compute the loss function
function loss = Loss(x, v)
    % v is of size (K, d, 2)
    % Initialize y
    [N, d] = size(x);
    K = size(v,1);
    y = zeros(N,K);
    
    % Loop through each element of y
    for i = 1:size(y,1) % Loop over rows of x
        for j = 1:size(y,2) % Loop over slices of v
    
            % Extract the ith row of x
            xi = x(i, :);
    
            % Extract the jth slice of v
            vj1 = v(j, :, 1);
            vj2 = v(j, :, 2);
    
            % Perform the calculation
            term = xi - dot(xi, vj1) * vj1 - dot(xi, vj2) * vj2;
            y(i, j) = 0.5 * norm(term)^2;
        end
    end
    loss = min(y, [], 2);
end


function loss = Loss_A(x, A)
    % A is of size (K, d, r)
    % x is of size (N, d)
    [N, d] = size(x);
    K = size(A,1);
    y = zeros(N,K);
    
    % Loop to calculate y(i,j)
    for i = 1:N
        for j = 1:K
            % Reshape A(j,:,:) into a 2D matrix of size d x r
            Aj = squeeze(A(j,:,:));

            % Compute the norm squared and multiply by 0.5
            y(i, j) = 0.5 * norm(x(i,:) * Aj)^2;
        end
    end
    loss = min(y, [], 2);
end



function loss_match = Loss_A_matching(x, A, mask)
    % A is of size (K, d, r)
    % x is of size (N, d)
    [N, d] = size(x);
    K = size(A,1);
    y = zeros(N,K);
    
    % Loop to calculate y(i,j)
    for i = 1:N
        for j = 1:K
            % Reshape A(j,:,:) into a 2D matrix of size d x r
            Aj = squeeze(A(j,:,:));

            % Compute the norm squared and multiply by 0.5
            y(i, j) = 0.5 * norm(x(i,:) * Aj)^2;
        end
    end
    [~, A_match] = min(y, [], 2);
    loss_match = minDistance(A_match, mask, K);
end



function y = Loss_A_detail(x, A)
    % A is of size (K, d, r)
    % x is of size (N, d)
    [N, d] = size(x);
    K = size(A,1);
    y = zeros(N,K);
    
    % Loop to calculate y(i,j)
    for i = 1:N
        for j = 1:K
            % Reshape A(j,:,:) into a 2D matrix of size d x r
            Aj = squeeze(A(j,:,:));

            % Compute the norm squared and multiply by 0.5
            y(i, j) = 0.5 * norm(x(i,:) * Aj)^2;
        end
    end
end



%% normal initialization
function A = normal_init(K, d, r)
    % Initialize A
    A = zeros(K, d, r);
    
    % Loop through each k
    for k = 1:K
        % Generate a random d x r matrix
        randMatrix = randn(d, r);
    
        % Perform QR decomposition
        [Q, ~] = qr(randMatrix, 0); % '0' option ensures Q has the same number of columns as randMatrix
    
        % Assign to A
        A(k,:,:) = Q(:, 1:r); % In case d > r, we only take the first r columns
    end
end

%% initialization using random index uniform sampling
function A = uniform_seeding(x, K, r)
    [N, d] = size(x);
    A = zeros(K, d, r);

    I_sampled = randperm(N, K);
    % Loop through each k
    for k = 1:K
        % Create a d x (r+1) matrix where the first column is aligned with x(I_sampled(k),:)
        V = zeros(d, r + 1);
        V(:,1) = x(I_sampled(k),:)';
    
        % Fill the rest of the columns with random values
        V(:,2:end) = randn(d, r);
    
        % Perform QR decomposition
        [Q, ~] = qr(V, 0); 
    
        % Use the last r columns of Q which are orthogonal to x(I_sampled(k),:)
        A(k,:,:) = Q(:, 2:end);
    end
end

%% K-means++ typed initializaiton
function A = careful_seeding(x, K, r)
    [N, d] = size(x);
    A = zeros(K, d, r);

    I_sampled = randi([1,N], 1, 1);  % uniformly sample the first token
    % Create a d x (r+1) matrix where the first column is aligned with x(I_sampled,:)
    V = zeros(d, r + 1);
    V(:,1) = x(I_sampled,:)';

    % Fill the rest of the columns with random values
    V(:,2:end) = randn(d, r);
    % Perform QR decomposition
    [Q, ~] = qr(V, 0); 
    % Use the last r columns of Q which are orthogonal to x(I_sampled,:)
    A(1,:,:) = Q(:, 2:end);

    for k = 2:K
        score = Loss_A_detail(x, A);
        score = min(score(:,1:k-1),2);
        weights = score / sum(score);
        I_sampled = randsample(N, 1, true, weights);
        % Create a d x (r+1) matrix where the first column is aligned with x(I_sampled,:)
        V = zeros(d, r + 1);
        V(:,1) = x(I_sampled,:)';
    
        % Fill the rest of the columns with random values
        V(:,2:end) = randn(d, r);
        % Perform QR decomposition
        [Q, ~] = qr(V, 0); 
        % Use the last r columns of Q which are orthogonal to x(I_sampled,:)
        A(k,:,:) = Q(:, 2:end);
    end
end

%% EM algorithm
function [A, loss] = BCD(x, A0, iter_max)
    A = A0;
    N = size(x, 1);
    [K, d, r] = size(A);
    
    Sigma = zeros(N, K);
    % Loop to calculate Sigma(i,j)
    for i = 1:N
        for j = 1:K
            % Reshape A(j,:,:) into a 2D matrix of size d x r
            Aj = squeeze(A(j,:,:));

            % Compute the squared norm
            Sigma(i, j) = norm(x(i,:) * Aj)^2;
        end
    end


    for lp = 1:iter_max
        for k = 1:K
            % update W
            W = ones(N, 1);
            % Loop to calculate W(i)
            for i = 1:N
                for j = 1:K
                    if j ~= k
                        W(i) = W(i) * Sigma(i, j);
                    end
                end
            end

            % update Ak
            M = x' * diag(W) * x;
            % Compute eigenvalues and eigenvectors
            [V, D] = eig(M);
        
            % Extract the eigenvalues
            eigenvalues = diag(D);
        
            % Sort the eigenvalues and get indices
            [~, indices] = sort(eigenvalues, 'ascend');
        
            % Select the first r indices
            r_indices = indices(1:r);
        
            % Update A(k,:,:) with the corresponding eigenvectors
            A(k,:,:) = V(:, r_indices);
            Ak = squeeze(A(k,:,:));
            
            % update Sigma
            for i = 1:N
                Sigma(i, k) = norm(x(i,:) * Ak)^2;
            end
        end
    end
    loss = sum(Loss_A(x, A));
end