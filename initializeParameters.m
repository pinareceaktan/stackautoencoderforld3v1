function theta = initializeParameters(hiddenSize, visibleSize)

%% Initialize parameters randomly based on layer sizes.
% r  = sqrt(6) / sqrt(hiddenSize+visibleSize+1);   % we'll choose weights uniformly from the interval [-r, r]
% W1 = rand(hiddenSize, visibleSize) * 2 * r - r; % hidden layer * input layer sized
% W2 = rand(visibleSize, hiddenSize) * 2 * r - r; % final layer * hidden layer sized

b1 = zeros(hiddenSize, 1); % input layer * 1 sized
b2 = zeros(visibleSize, 1); % hidden layer *1 sized

W1 = normrnd(0,0.003,[hiddenSize visibleSize]);
W2 = normrnd(0,0.003,[visibleSize hiddenSize]);




% Convert weights and bias gradients to the vector form.
% This step will "unroll" (flatten and concatenate together) all 
% your parameters into a vector, which can then be used with minFunc. 
theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];

end

