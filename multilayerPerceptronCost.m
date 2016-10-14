%% TRHRASH CODE 
% function [cost,grad] = multilayerPerceptronCost(theta, visibleSize,hiddenSize,...
%     lambda,data,yi)
% W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
% W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
% b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
% b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);
% 
% Theta{1,1} = W1;
% Theta{1,2} = W2;
% 
% m = size(data,2);
% nl = 3; % number of layers
% %% Forward Propagation 
% bias = b1;
% a(1) = {data}; % input
% for i = 2:nl       % loop through hidden layers 
%     l = i ; % next layer
%     lpre = i-1; % previous layer
%     z(l) =  {Theta{1,lpre}*a{1,lpre}+repmat(bias,1,m)};
%     a(l)=    {sigmoid(z{1,l})};
%     bias = b2 ;
% end
% cost_err= sum(sum((yi-a{1,1}).*(yi-a{1,1})))/2;% J(w,b) cost
% 
% %% Back Propagation
% for i = nl:-1:2 % back looping through layers
%     lb = i ;
%     if lb == nl % is it the output layer 3.layer
%         delta(1,lb)= {(yi-a{1,3}).*sigmoidinv(z{1,lb})} ;
%     else  % hidden layer  
%         delta(1,lb) = {(Theta{1,lb}'*delta{1,lb+1}).*sigmoidinv(z{1,lb})};
%     end
%      % Compute the derivatives
%     partialw(lb) = {delta{1,lb}*a{lb-1}'};
%     partialb(lb-1) = {sum(delta{1,lb},2)};
% end
% 
% W1grad = partialw{1,2}*1/m+lambda*Theta{1,1};
% b1grad =  1/m*partialb{1,1};
% W2grad = partialw{1,3}*1/m+lambda*Theta{1,2};
% b2grad = 1/m*partialb{1,2};
% 
% cost_err = cost_err/m;
% cost_weights = lambda/2*(sum(W1(:).^2) + sum(W2(:).^2)); % w regularization weight decay parameter
% cost = cost_err + cost_weights ;
% 
% grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
% 
% end
% 
% function sigm = sigmoid(x)
%   
%     sigm = 1 ./ (1 + exp(-x));
% end
% function inv = sigmoidinv(z)
%     inv = z.*(1-z);
% end
