function [cost,grad] = sparseAutoencoderCost(theta, inputSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% inputSize:(visibleSize) the number of input units 
% hiddenSize: the number of hidden units
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
%% Unroll Parameters
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 

Theta{1,1} = reshape(theta(1:hiddenSize*inputSize), hiddenSize, inputSize);
Theta{1,2} = reshape(theta(hiddenSize*inputSize+1:2*hiddenSize*inputSize), inputSize, hiddenSize);
b{1,1} = theta(2*hiddenSize*inputSize+1:2*hiddenSize*inputSize+hiddenSize);
b{1,2} = theta(2*hiddenSize*inputSize+hiddenSize+1:end);

%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
%
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 

m = size(data,2); % number of train samples
nl = 3 ;          % number of layers
sc =  sparsityParam; 

%% Forward propogation
a(1) = {data}; % input
for i = 2:nl %loop through the hidden layer
    l = i ; % next layer
    lpre = i-1; % previous layer
    z(l) =  {Theta{1,lpre}*a{1,lpre}+repmat(b{1,lpre},1,m)};
    a(l)=    {sigmoid(z{1,l})};
end
rho = sum(a{1,2},2)/m; % mean of the activations in hidden layer, sum over all train samples
cost_err= 0.5*sumsqr(a{1,3}-a{1,1});% J(w,b) cost

%% Backpropagation : Error of node j in layer l
for i = nl:-1:2 % for each layer back to the front
    lb = i;
    if lb == nl % is it the output layer 3.layer
        delta(1,lb)= {(a{1,3}-a{1,1}).*sigmoidinv(a{1,lb})} ;
    else
        % hidden layer              
        delta(1,lb) = {(Theta{1,lb}'*delta{1,lb+1}+repmat(beta*(-(sc./rho)+(1-sc)./(1-rho)),1,m)).*sigmoidinv(a{1,lb})};
    end
    
    % Compute the derivatives
    partialw(lb) = {delta{1,lb}*a{lb-1}'};
    partialb(lb-1) = {sum(delta{1,lb},2)};

end
%% Gradient descent to decrease the cost function

W1grad = partialw{1,2}*1/m+lambda*Theta{1,1};
b1grad =  1/m*partialb{1,1};
W2grad = partialw{1,3}*1/m+lambda*Theta{1,2};
b2grad = 1/m*partialb{1,2};


%% Regularization
KLdiv = sc*log(sc./rho) + (1 - sc)*log((1 - sc)./(1 - rho));

cost_err = cost_err/m;
cost_weights = lambda/2*(sum(Theta{1,1}(:).^2) + sum(Theta{1,2}(:).^2)); % w regularization
cost_sparse = beta*sum(KLdiv); % induce "sparsity"

cost = cost_err + cost_weights + cost_sparse;



grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end


function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end
function inv = sigmoidinv(z)
    inv = z.*(1-z);
end
