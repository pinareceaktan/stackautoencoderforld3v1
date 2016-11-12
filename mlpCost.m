function [cost,grad] = mlpCost(theta,inputSize,outputSize,lambda,data,groundTruth)

%% Unroll The Parameters 
Theta{1,1} = reshape(theta(1:outputSize*inputSize), outputSize, inputSize);
b{1,1} = theta(outputSize*inputSize+1:end);
% W1 : 400*136
% b1 : 136*1
 
m = size(data,2); % number of train sampes
nl = 2; % number of layers 

%% Forward Propagation
a(1) = {data};
for i = 2: nl % loop through hidden layers
    l= i; % next layer
    lpre = i-1; % previous layer
    z(l) =  {Theta{1,lpre}*a{1,lpre}+repmat(b{1,lpre},1,m)};
    a(l)=    {sigmoid(z{1,l})};
end

%% Cost Function
cost_err= 1/m*0.5*sumsqr(a{1,nl}-groundTruth);% J(w,b) cost

% Calculating regularization term 
for j = 1: nl-1 % networkde kaç tane layer varsa 
    sumup{1,j} = sum(Theta{1,j}(:).^2);
end

regularization_term = lambda/2*sum(cell2mat(sumup));
cost = cost_err + regularization_term;


%% Back Prop.
 
for i = nl:-1:2 % for each layer back to the front
%     disp(['layer: ' num2str(i)])
    
    if i == nl % is it the output layer 
        % output layer
        delta{i} = (-1*(groundTruth-a{1,nl})).*sigmoidinv(a{1,i}); % hadamard product between sigmoid inv and error
    else
        % hidden layer
        delta{i} = (Theta{1,i}'*delta{1,i+1}).* sigmoidinv(a{1,i});
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

grad = [W1grad(:) ; b1grad(:)];

end
function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end
function inv = sigmoidinv(z)
    inv = z.*(1-z);
end