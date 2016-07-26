function [C, sigma] = dataset3Params(X, y, Xval, yval, do_search)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%


if ~exist('do_search', 'var') || isempty(do_search)
    do_search = false;
end


% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

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

if do_search

    choices = [0.01 0.03 0.1 0.3 1 3 10 30];

    best = inf;

    % SVM Parameters
    for C_idx = 1:size(choices,2)
        for sigma_idx = 1:size(choices,2)
            curr_C = choices(C_idx);
            curr_sigma = choices(sigma_idx);

            fprintf("Running for C=%f, sigma=%f\n", curr_C, curr_sigma);

            % We set the tolerance and max_passes lower here so that the code will run
            % faster. However, in practice, you will want to run the training to
            % convergence.
            model= svmTrain(X, y, curr_C, @(x1, x2) gaussianKernel(x1, x2, curr_sigma)); 
            predictions = svmPredict(model, Xval);
            error = mean(double(predictions ~= yval))

            if error < best
                best = error
                C = curr_C
                sigma = curr_sigma
            endif

        end
    end
    fprintf("Best error is %f with C=%f, sigma=%f\n", best, C, sigma);
else
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval))
    fprintf("Stored error is %f with C=%f, sigma=%f\n", error, C, sigma);
end

% =========================================================================

end
