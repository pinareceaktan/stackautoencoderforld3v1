function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses,perceptronSize, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% outputSize : visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer* 7 or your last
% year of the autoencoder
% numClasses:  the number of categories : 136 coordinates
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
perTheta{1,1} = reshape(theta(1:perceptronSize*hiddenSize), perceptronSize, hiddenSize);
perTheta{1,2} = reshape(theta(perceptronSize*hiddenSize+1:perceptronSize*hiddenSize+1+perceptronSize*numClasses-1), numClasses, perceptronSize);
perb{1,1} = theta(perceptronSize*hiddenSize+1+perceptronSize*numClasses:perceptronSize*hiddenSize+1+perceptronSize*numClasses+perceptronSize-1);
perb{1,2} = theta(perceptronSize*hiddenSize+1+perceptronSize*numClasses+perceptronSize:perceptronSize*hiddenSize+perceptronSize*numClasses+perceptronSize+numClasses);

% Extract out the "stack"
stack = params2stack(theta(perceptronSize*hiddenSize+perceptronSize*numClasses+perceptronSize+numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

% feedforward pass over autoencoders
depth = numel(stack);
z = cell(depth+1,1);
a = cell(depth+1,1);
a{1} = double(data);
for l=1:depth,
    wa = stack{l}.w*a{l};
    z{l+1} = bsxfun(@plus, wa, stack{l}.b);
    a{l+1} = sigmoid(z{l+1});
end

%% Feedforward over multi layer neural network
nl = 3;
ap(1) = {double(a{depth+1})}; % auto encoderdan çýkan featurelar
for i = 2: nl % loop through hidden layers
     l= i; % next layer
     lpre = i-1; % previous layer
     zp(l) =  {perTheta{1,lpre}*ap{1,lpre}+repmat(perb{1,lpre},1,size(data,2))};
     ap(l)=    {sigmoid(zp{1,l})};
end
 pred = ap{1,nl}*100; % denormalization


% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
