function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, groundTruth,perceptronSize)
                                         
% stackedAECost: Takes a trained multilayer neural network and a training data set with groundTruths,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% groundTruths: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll Classifier Parameters : Perceptron 
% We first extract the part which compute the multi layer network gradient
perTheta{1,1} = reshape(theta(1:perceptronSize*hiddenSize), perceptronSize, hiddenSize);
perTheta{1,2} = reshape(theta(perceptronSize*hiddenSize+1:perceptronSize*hiddenSize+1+perceptronSize*numClasses-1), numClasses, perceptronSize);
perb{1,1} = theta(perceptronSize*hiddenSize+1+perceptronSize*numClasses:perceptronSize*hiddenSize+1+perceptronSize*numClasses+perceptronSize-1);
perb{1,2} = theta(perceptronSize*hiddenSize+1+perceptronSize*numClasses+perceptronSize:perceptronSize*hiddenSize+perceptronSize*numClasses+perceptronSize+numClasses);

% Extract out the "stack"  parameters
stack = params2stack(theta(perceptronSize*hiddenSize+perceptronSize*numClasses+perceptronSize+numClasses+1:end), netconfig);
%% buraya kadar okey
% You will need to compute the following gradients
% perceptronThetaGrad1 = cell(size(perTheta{1,2}));
% perceptronThetaGrad1 = cell(size(perTheta{1,2}));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end


cost = 0; % You need to compute this

% You might find these variables useful
m = size(data, 2);

%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

% feedforward pass over autoencoders
depth = numel(stack); % depth of stack
z = cell(depth+1,1);
a = cell(depth+1,1);
a{1} = data;
for l=1:depth,
    wa = stack{l}.w*a{l};
    z{l+1} = bsxfun(@plus, wa, stack{l}.b);
    a{l+1} = sigmoid(z{l+1});
end

%% Feedforward over multi layer neural network
nl = 3;
ap(1) = {a{depth+1}}; % auto encoderdan çýkan featurelar
for i = 2: nl % loop through hidden layers
     l= i; % next layer
     lpre = i-1; % previous layer
     zp(l) =  {perTheta{1,lpre}*ap{1,lpre}+repmat(perb{1,lpre},1,size(data,2))};
     ap(l)=    {sigmoid(zp{1,l})};
 end
h = ap{1,nl};
%% cost err of perceptron 
cost_err = 0.5 * sumsqr(h-groundTruth);
%% Back Prop.
for i = nl:-1:2
    lb = i;
     if lb == nl % is it the output layer 3.layer
        perdelta(1,lb)= {(ap{1,lb}-groundTruth).*sigmoidinv(ap{1,lb})} ;
    else  % hidden layer  
        perdelta(1,lb) = {(perTheta{1,lb}'*perdelta{1,lb+1}).*sigmoidinv(ap{1,lb})};
     end
    perpartialw(lb) = {perdelta{1,lb}*ap{lb-1}'};
    perpartialb(lb-1) = {sum(perdelta{1,lb},2)};
 end
% Gradients 
W1grad = perpartialw{1,2}*1/m+lambda*perTheta{1,1};
b1grad =  1/m*perpartialb{1,1};
W2grad = perpartialw{1,3}*1/m+lambda*perTheta{1,2};
b2grad = 1/m*perpartialb{1,2};
% Cost
cost_err = cost_err/m;
cost_weights = lambda/2*(sum(perTheta{1,1}(:).^2) + sum(perTheta{1,2}(:).^2)); % w regularization weight decay parameter
cost = cost_err + cost_weights ;
%% FINE TUNE!    
% deltas. Note that sparsityParam is not used for fine tuning
delta = cell(depth+1);
delta{depth+1} = -(W1grad'*W2grad'*(groundTruth-h)) .* a{depth+1};

for l=depth:-1:2,
    delta{l} = (stack{l}.w'*delta{l+1}) .* a{l};
end

for l=depth:-1:1,
    stackgrad{l}.w = delta{l+1}*a{l}'/m;
    stackgrad{l}.b = sum(delta{l+1},2)/m;
end






%% Roll gradient vector
perceptronThetaGrad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

grad = [perceptronThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function inv = sigmoidinv(z)
    inv = z.*(1-z);
end
