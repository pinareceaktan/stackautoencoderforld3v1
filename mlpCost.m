function [cost,grad] = mlpCost(theta,inputSize,hiddenSize,outputSize,lambda,data,groundTruth)

%% Unroll The Parameters 
Theta{1,1} = reshape(theta(1:hiddenSize*inputSize), hiddenSize, inputSize);
Theta{1,2} = reshape(theta(hiddenSize*inputSize+1:hiddenSize*inputSize+1+hiddenSize*outputSize-1), outputSize, hiddenSize);
b{1,1} = theta(hiddenSize*inputSize+1+hiddenSize*outputSize:hiddenSize*inputSize+1+hiddenSize*outputSize+hiddenSize-1);
b{1,2} = theta(hiddenSize*inputSize+1+hiddenSize*outputSize+hiddenSize:end);
% W1 : 200*10000
% W2 : 136*200
% b1 : 200*1
% b2 : 136*1
 
m = size(data,2); % number of train sampes
nl = 3; % number of layers 
%% Forward Propagation
a(1) = {data};
for i = 2: nl % loop through hidden layers
     l= i; % next layer
     lpre = i-1; % previous layer
     z(l) =  {Theta{1,lpre}*a{1,lpre}+repmat(b{1,lpre},1,m)};
     a(l)=    {sigmoid(z{1,l})};
 end
%% Cost Err
cost_err = 0.5 * sumsqr((a{1,nl}-groundTruth));
%% Back Prop.
for i = nl:-1:2
    lb = i;
     if lb == nl % is it the output layer 3.layer
        delta(1,lb)= {(a{1,lb}-groundTruth).*sigmoidinv(a{1,lb})} ;
    else  % hidden layer  
        delta(1,lb) = {(Theta{1,lb}'*delta{1,lb+1}).*sigmoidinv(a{1,lb})};
     end
    partialw(lb) = {delta{1,lb}*a{lb-1}'};
    partialb(lb-1) = {sum(delta{1,lb},2)};
 end

W1grad = partialw{1,2}*1/m+lambda*Theta{1,1};
b1grad =  1/m*partialb{1,1};
W2grad = partialw{1,3}*1/m+lambda*Theta{1,2};
b2grad = 1/m*partialb{1,2};

cost_err = cost_err/m;
cost_weights = lambda/2*(sum(Theta{1,1}(:).^2) + sum(Theta{1,2}(:).^2)); % w regularization weight decay parameter
cost = cost_err + cost_weights ;

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
end
function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end
function inv = sigmoidinv(z)
    inv = z.*(1-z);
end