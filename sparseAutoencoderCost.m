function [cost,grad] = sparseAutoencoderCost(theta, inputSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)
% inputSize     :(visibleSize) the number of input units 
% hiddenSize    : the number of hidden units
% lambda        : weight decay parameter
% sparsityParam : The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term

% Unroll Parameters
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 

Theta{1,1} = reshape(theta(1:hiddenSize*inputSize), hiddenSize, inputSize);
Theta{1,2} = reshape(theta(hiddenSize*inputSize+1:2*hiddenSize*inputSize), inputSize, hiddenSize);
b{1,1} = theta(2*hiddenSize*inputSize+1:2*hiddenSize*inputSize+hiddenSize);
b{1,2} = theta(2*hiddenSize*inputSize+hiddenSize+1:end);

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

% Back Propagation step 1: Forward Pass
a(1) = {data}; % input
for i = 2:nl %loop through the hidden layer
    l = i ; % next layer
    lpre = i-1; % previous layer
    z(l) =  {Theta{1,lpre}*a{1,lpre}+repmat(b{1,lpre},1,m)};
    a(l)=    {sigmoid(z{1,l})};
end

% Cost Function
cost_err= 1/m*0.5*sumsqr(a{1,3}-a{1,1});% J(w,b) cost
% Calculating regularization term : dummy but accurate
for j = 1:nl-1 % networkde kaç layer varsa
    number_of_units = numel(Theta{1,j});
    sumup{1,j} = 0;
    for i = 1: number_of_units
        sumup{1,j} = sumup{1,j} + Theta{1,j}(i)^2;
    end
end
% Calculating regularization term : smart but not accurate
% for j = 1: nl-1 % networkde kaç tane layer varsa 
%     sum{1,j} = sum(Theta{1,j}(:).^2);
% end
disp('dssad');
regularization_term = lambda/2*sum(cell2mat(sumup));
cost_err = cost_err + regularization_term;

%  Back Propagation step 2 : Error of node j in layer l
for i = nl:-1:2 % for each layer back to the front
    lb = i;
    if lb == nl % is it the output layer 
        % output layer
        for j = 1: numel(a{1,nl}) % output layerdaki her nöron için
          errors_per_unit(lb,j) = (-1*(a{1,1}(j)-a{1,3}(j)))*sigmoidinv(z{1,lb}(j));
        end
    else
        % hidden layer
        for k = 1: numel(a{1,i}) % mevcut hidden layerdaki her nöron için
            for j = 1: numel(a{1,i+1}) % bir sonraki layerdaki nöron sayýsý kadar dön
            errors_per_unit(i,k) = (Theta{1,i}(i,k)*errors_per_unit(i+1,j))* sigmoidinv(z{1,i}(k));
            end
        end
%         delta(1,lb) = {(Theta{1,lb}'*delta{1,lb+1}+repmat(beta*(-(sc./rho)+(1-sc)./(1-rho)),1,m)).*sigmoidinv(a{1,lb})};
    end
    
    % Compute the derivatives
    partialw(lb) = {delta{1,lb}*a{lb-1}'};
    partialb(lb-1) = {sumup(delta{1,lb},2)};

end
% Gradient descent to decrease the cost function

W1grad = partialw{1,2}*1/m+lambda*Theta{1,1};
b1grad =  1/m*partialb{1,1};
W2grad = partialw{1,3}*1/m+lambda*Theta{1,2};
b2grad = 1/m*partialb{1,2};






grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end


% cost_sparse = beta*sum(KLdiv); % induce "sparsity"
% 
% cost = cost_err + cost_weights + cost_sparse;
% 
% % Regularization
% rho = sum(a{1,2},2)/m; % mean of the activations in hidden layer, sum over all train samples
% KLdiv = sc*log(sc./rho) + (1 - sc)*log((1 - sc)./(1 - rho));
function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end
function inv = sigmoidinv(z)
    inv = z.*(1-z);
end
