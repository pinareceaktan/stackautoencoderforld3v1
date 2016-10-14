function [activation] = feedforwardperceptron(theta,hiddenSize,inputSize,outputSize,data)
% Unrolling parameters
Theta{1,1} = reshape(theta(1:hiddenSize*inputSize), hiddenSize, inputSize);
Theta{1,2} = reshape(theta(hiddenSize*inputSize+1:hiddenSize*inputSize+1+hiddenSize*outputSize-1), outputSize, hiddenSize);
b{1,1} = theta(hiddenSize*inputSize+1+hiddenSize*outputSize:hiddenSize*inputSize+1+hiddenSize*outputSize+hiddenSize-1);
b{1,2} = theta(hiddenSize*inputSize+1+hiddenSize*outputSize+hiddenSize:end);
nl = 3;

a(1) = {double(data)};
for i = 2: nl % loop through hidden layers
     l= i; % next layer
     lpre = i-1; % previous layer
     z(l) =  {Theta{1,lpre}*a{1,lpre}+repmat(b{1,lpre},1,size(data,2))};
     a(l)=    {sigmoid(z{1,l})};
 end
% minVal = min(a{1,1});
% maxVal = max(a{1,1});
% activation = minVal + a{1,1}.*(maxVal - minVal)

activation = a{1,nl}*100

end

function sigm = sigmoid(x)
%     digits(15)
%     sigm = vpa(1 ./ (1 + exp(-x)));
        sigm = (1 ./ (1 + exp(-x)));

end