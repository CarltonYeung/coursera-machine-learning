function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % for-loop implementation
    %{
    delta = zeros(rows(theta), 1);
    for ex = 1:m
        x_ex = X(ex, :);
        hypothesis = x_ex * theta;
        diff = hypothesis - y(ex);
        adjustment = diff * x_ex';
        delta = delta + adjustment;
    end
    delta = delta / m;

    theta = theta - alpha * delta;
    %}

    % vectorized implementation
    hypothesis = X * theta;         % (m x j) * (j x 1) = (m x 1)
    diff = hypothesis - y;          % (m x 1) - (m x 1) = (m x 1)
    delta = X' * diff;              % (j x m) * (m x 1) = (j x 1)
    delta = delta / m;              % (j x 1) / scalar = (j x 1)
    theta = theta - alpha * delta;  % (j x 1) - scalar * (j x 1) = (j x 1)

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
