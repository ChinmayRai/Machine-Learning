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

k=sigmoid((theta'*X')');

thetaorg=theta(2:end);

J=((-y)'*log(k)+(y-1)'*log(1-k))*(1/m)+(lambda/(2*m))*(thetaorg'*thetaorg);

grad=(((k-y)'*X)')./m +(lambda/m)*[0;thetaorg];



end
