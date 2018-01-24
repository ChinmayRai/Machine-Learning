function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%


C_val=[0.01,0.03,0.1,0.3,1,3,10,30];                 %i
sigma_val=[0.01,0.03,0.1,0.3,1,3,10,30];             %j
accuracy=zeros(8,8);
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))

for i=1:8
  for j=1:8
    model= svmTrain(X, y, C_val(i), @(x1, x2) gaussianKernel(x1, x2, sigma_val(j)));
    predictions=svmPredict(model, Xval);   accuracy(i,j)=mean(double(predictions ~= yval));
  end;
end;

[e,i_val]=min(accuracy);
[f,j]=min(e);
i=i_val(j);
C=C_val(i);
sigma=sigma_val(j);

end
