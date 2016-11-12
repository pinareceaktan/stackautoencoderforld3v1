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

% Calculating regularization term : dummy but accurate in e-11 order
        % for j = 1:nl-1 % networkde kaç layer varsa
        %     number_of_units = numel(Theta{1,j});
        %     sumup{1,j} = 0;
        %     for i = 1: number_of_units
        %         sumup{1,j} = sumup{1,j} + Theta{1,j}(i)^2;
        %     end
        % end
% Calculating regularization term 
for j = 1: nl-1 % networkde kaç tane layer varsa 
    sumup{1,j} = sum(Theta{1,j}(:).^2);
end

regularization_term = lambda/2*sum(cell2mat(sumup));
cost_err = cost_err + regularization_term;

% Calculating sparsity parameter
rho_hat = sum(a{1,2},2)/m;
sparsity_deriv = beta*...
    (-sparsityParam./rho_hat + (1-sparsityParam)./(1-rho_hat));

%  Back Propagation step 2 : Error of node j in layer l
% Dummy way to compute gradients
    % for i = nl:-1:2 % for each layer back to the front
    %     disp(['layer: ' num2str(i)])
    %     lb = i;
    %     if lb == nl % is it the output layer 
    %         % output layer
    %         for j = 1: numel(a{1,nl}) % output layerdaki her nöron için
    %           disp(['in output layer, node: ' num2str(j)]);
    %           errors_per_unit(lb,j) = (-1*(a{1,1}(j)-a{1,3}(j)))*sigmoidinv(a{1,lb}(j));
    %         end
    %     else
    %         % hidden layer
    %         for k = 1: numel(a{1,i}) % mevcut hidden layerdaki her nöron için
    %             disp(['in layer ' num2str(i) ' , node: ' num2str(k)]);
    %             for j = 1: numel(a{1,i+1}) % bir sonraki layerdaki nöron sayýsý kadar dön
    %             disp(['in layer ' num2str(i+1) ' , node: ' num2str(j)]);    
    %             errors_per_unit(i,k) = (Theta{1,i}(j,k)*errors_per_unit(i+1,j))* sigmoidinv(a{1,i}(k));
    %             end
    %         end
    % %         delta(1,lb) = {(Theta{1,lb}'*delta{1,lb+1}+repmat(beta*(-(sc./rho)+(1-sc)./(1-rho)),1,m)).*sigmoidinv(a{1,lb})};
    %     end
    % end
% Smart vectorized version
for i = nl:-1:2 % for each layer back to the front
%     disp(['layer: ' num2str(i)])
    
    if i == nl % is it the output layer 
        % output layer
        delta{i} = (-1*(a{1,1}-a{1,3})).*sigmoidinv(a{1,i}); % hadamard product between sigmoid inv and error
    else
        % hidden layer
        delta{i} = (Theta{1,i}'*delta{1,i+1}+ repmat(sparsity_deriv,1,m)).* sigmoidinv(a{1,i});
    end
end

% Computing partial derivatives

for l = 1:nl-1
    partial_weights(l) = {delta{l+1}*a{1,l}'};
    partial_biases(l)  = {delta{l+1}};
end


% Computing gradients 

W1grad = partial_weights{1,1}*1/m+lambda*Theta{1,1};
b1grad =  mean(partial_biases{1,1},2);
W2grad = partial_weights{1,2}*1/m+lambda*Theta{1,2};
b2grad = mean(partial_biases{1,2},2);

% KL divergence
KLdiv = sparsityParam*log(sparsityParam./rho_hat) + ...
    (1 - sparsityParam)*log((1 - sparsityParam)./(1 - rho_hat));

cost_sparse = beta*sum(KLdiv); % induce "sparsity"
cost = cost_err  + cost_sparse;



grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end
function inv = sigmoidinv(z)
    inv = z.*(1-z);
end
