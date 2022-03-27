function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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
%

C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
C_len = size(C_values, 2);
sigma_values = C_values;
sigma_len = size(sigma_values, 2);
J_val = zeros([C_len, sigma_len]);

for c_index = 1:C_len
    for sigma_index = 1:sigma_len
        C_val = C_values(c_index);
        sigma_val = sigma_values(sigma_index);

        model = svmTrain(X, y, C_val, @(x1, x2) gaussianKernel(x1, x2, sigma_val));
        predictions = svmPredict(model, Xval);

        J_val(c_index, sigma_index) = mean(double(predictions ~= yval));
    endfor
endfor

J_val

J_min = J_val(1, 1);
C = C_values(1);
sigma = sigma_values(1);

for c_index = 1:C_len
    for sigma_index = 1:sigma_len
        if J_val(c_index, sigma_index) < J_min
            C = C_values(c_index);
            sigma = sigma_values(sigma_index);
            J_min = J_val(c_index, sigma_index);
        endif
    endfor
endfor

C
sigma

% =========================================================================

end
