function [landmark_predictions,pose_predictions] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% outputSize : visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 3rd layer* Your
% autoencoder's last layer
% numClasses:  the number of categories : 136 coordinates + 2 pose classes
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

 
%% Unroll theta parameter

% We first extract the part which compute perceptron gradient
perTheta{1,1} = reshape(theta(1:numClasses*hiddenSize), numClasses, hiddenSize); % 138x400
perb{1,1} = theta(numClasses*hiddenSize+1:numClasses*hiddenSize+numClasses);   % 138x1


% Extract out the "stack"
stack = params2stack(theta(numClasses*hiddenSize+numClasses+1:end), netconfig);


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
nl = 2;
ap(1) = {double(a{depth+1})}; % auto encoderdan çýkan featurelar
for i = 2: nl % loop through hidden layers
     l= i; % next layer
     lpre = i-1; % previous layer
     zp(l) =  {perTheta{1,lpre}*ap{1,lpre}+repmat(perb{1,lpre},1,size(data,2))};
     ap(l)=    {sigmoid(zp{1,l})};
end
% Renormalize ap
normalized_predictions = reshape(im2double(ap{1,nl}(:)),size(ap{1,2},1),size(ap{1,2},2));
% Denormalize ap
landmark_predictions = normalized_predictions{1,nl}(1:136,:)*50; % denormalization
pose_predictions     =   normalized_predictions{1,nl}(137:end,:);

% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
