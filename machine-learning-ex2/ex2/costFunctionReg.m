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

% Cost
J = 1 / m * sum(-y.*log(h) - (1-y).*log(1 - h));
% Add the regularization term
J += lambda / 2 / m * (sum(theta' * theta) - theta(1)^2);

% Gradient
grad = 1 / m .* sum(repmat((h-repmat(y, 1, size(h,2))), 1, size(X,2)) .* X);
% Add the regularization term (but not for theta(1))
reg = lambda / m * theta;
reg(1) = 0;
grad += reg';

% =============================================================

end
