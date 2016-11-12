function theta = initializePerceptronParams(outputSize,inputSize)
%% Initialize parameters randomly based on layer sizes.
% Not a normal distribution, implementation using hidden layer

% r  = sqrt(6) / sqrt(outputSize+hiddenSize+inputSize+1);   % we'll choose weights uniformly from the interval [-r, r]
% W1 = rand(hiddenSize, inputSize) * 2 * r - r; % hidden layer * input layer sized
% W2 = rand(outputSize, hiddenSize) * 2 * r - r; % final layer * hidden layer sized
% b1 = zeros(hiddenSize, 1); % input layer * 1 sized
% b2 = zeros(outputSize, 1); % hidden layer *1 sized
% theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];

% normal distribution on weights
W1 = normrand(0,0.003,[outputSize, inputSize]); % hidden layer * input layer sized
b1 = zeros(outputSize, 1); % hidden layer *1 sized

theta = [W1(:) ; b1(:) ];

end