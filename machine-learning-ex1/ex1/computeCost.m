function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% for-loop implementation
%{
for i = 1:m
  hypothesis = theta' * X(i, :)';
  diff = hypothesis - y(i);
  J = J + diff ^ 2;
end
J = J / (2 * m);
%}

% vectorized implementation
hypothesis = X * theta;  % m x 1 vector
cost = hypothesis - y;
cost_sq = cost .^ 2;
J = sum(cost_sq) / (2 * m);


% =========================================================================

end
