function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.


m = length(y); 

J = 0;

grad = zeros(size(theta));

k=sigmoid((theta'*X')');

J=((-y)'*log(k)+(y-1)'*log(1-k)).*(1/m);

grad=(((k-y)'*X)')/m;

end
