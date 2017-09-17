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

    % Dimensions
    % ==========
    % X = 97x2
    % y = 97x1
    % theta = 2x1
    % hypothesis = 97x1
 
    % Lectures : hypothesis = theta * X
    % DIM 97x1 = 97x2 * 2x1 
    hypothesis = X * theta;
    % Lectures : delta = (1/m)*((hypothesis - y)* X')
    % DIM 2x1 = n * 2x97 * (97x1 - 97x1) 
    delta =  (1/m) * X' * ((hypothesis - y));
    % Lectures : theta = theta - alpha * delta
    % DIM 2x1 = 2x1 - n * 2x1
    theta = theta - alpha * delta;
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
