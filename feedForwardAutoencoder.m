function [activation] = feedForwardAutoencoder(theta, hiddenSize, visibleSize, data)

% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize: the number of hidden units
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
  
% Unrolling parameters
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the activation of the hidden layer for the Sparse Autoencoder.

activation = sigmoid(W1*data + repmat(b1,1,size(data,2)));

end

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
