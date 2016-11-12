function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, groundTruth)
                                         
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
perTheta{1,1} = reshape(theta(1:numClasses*hiddenSize), numClasses, hiddenSize);
perb{1,1} = theta(numClasses*hiddenSize+1:numClasses*hiddenSize+1+numClasses-1);


% Extract out the "stack"  parameters
% stack = params2stack(theta(perceptronSize*hiddenSize+perceptronSize*numClasses+perceptronSize+numClasses+1:end), netconfig);
stack = params2stack(theta(numClasses*hiddenSize+numClasses+1:end), netconfig);

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

m = size(data, 2);

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

% feedforward over perceptron
nl = 2; % number of layers 
ap(1) = {a{depth+1}};
for i = 2: nl % loop through hidden layers
    l= i; % next layer
    lpre = i-1; % previous layer
    zp(l) =  {perTheta{1,lpre}*ap{1,lpre}+repmat(perb{1,lpre},1,m)};
    ap(l)=    {sigmoid(zp{1,l})};
end

% Cost Function : Perceptron
cost_err= 1/m*0.5*sumsqr(ap{1,nl}-groundTruth);% J(w,b) cost

% Calculating regularization term 
for j = 1: nl-1 % networkde kaç tane layer varsa 
    sumup{1,j} = sum(perTheta{1,j}(:).^2);
end

regularization_term = lambda/2*sum(cell2mat(sumup));
cost = cost_err + regularization_term;

%% Back Prop.
 
for i = nl:-1:2 % for each layer back to the front
    
    if i == nl % is it the output layer 
        % output layer
        perdelta{i} = (-1*(groundTruth-ap{1,3})).*sigmoidinv(ap{1,i}); % hadamard product between sigmoid inv and error
    else
        % hidden layer
        perdelta{i} = (perTheta{1,i}'*perdelta{1,i+1}).* sigmoidinv(ap{1,i});
    end
end

% Computing partial derivatives

for l = 1:nl-1
    perpartialw(l) = {perdelta{l+1}*ap{1,l}'};
    perpartialb(l)  = {perdelta{l+1}};
end

% Computing gradients 

W1grad = perpartialw{1,1}*1/m+lambda*perTheta{1,1};
b1grad =  mean(perpartialb{1,1},2);



%% FINE TUNE!    
% deltas. Note that sparsityParam is not used for fine tuning
delta = cell(depth+1);
delta{depth+1} = -(W1grad'*(groundTruth-h)) .* a{depth+1};

for l=depth:-1:2,
    delta{l} = (stack{l}.w'*delta{l+1}) .* a{l};
end

for l=depth:-1:1,
    stackgrad{l}.w = delta{l+1}*a{l}'/m;
    stackgrad{l}.b = sum(delta{l+1},2)/m;
end


%% Roll gradient vector
% perceptronThetaGrad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
perceptronThetaGrad = [W1grad(:) ; b1grad(:) ];

grad = [perceptronThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function inv = sigmoidinv(z)
    inv = z.*(1-z);
end
