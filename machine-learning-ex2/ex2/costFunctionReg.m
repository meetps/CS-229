function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h = sigmoid(X * theta);
J = (-y' * log(h) - (1 - y)' * log(1 - h)) * (1/m) + lambda / 2 * (1/m) * sum(theta(2:size(theta)).^2);
nonRegularized = X' * (h - y) * (1/m);
grad(1) =  nonRegularized(1);
grad(2:size(grad)) = nonRegularized(2:size(grad)) + lambda * (1/m) * theta(2:size(theta));

% =============================================================

end
