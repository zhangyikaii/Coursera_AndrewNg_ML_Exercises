function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


grad = zeros(size(theta));

hTheta = sigmoid(X * theta);

J = sum(log(hTheta)' * (-y) - log(1 - hTheta)' * (1 - y)) / m ...
    + (theta' * theta - theta(1, 1) * theta(1, 1)) * lambda / 2 / m;

grad(1, 1) = (hTheta - y)' * X(:, 1) / m;
for j = 2 : size(grad, 1)
    grad(j, 1) = (hTheta - y)' * X(:, j) / m + lambda / m * theta(j, 1);
end





% =============================================================

end
