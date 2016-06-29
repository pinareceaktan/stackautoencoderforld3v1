%%  Stacked Autoencoder implementation based on the courses of NG in CS294A/CS294W
%% STEP 0: Parameter initialization
inputSize       = 100 * 100;% image size
outputSize      = 68*2;     % 136 coordinates will be extracted
hiddenSizeL1    = 500;      % auto encoder extracts 500 features 
perceptronSize  = 200;      % perceptron has 200 neurons
hiddenSizeL2 =  300; 
sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-3;     % weight decay parameter       
beta = 3;              % weight of sparsity penalty term       

%% STEP 1: Load data from the MultiPie database
%% Uncomment if you fetch data set in meanwhile 
[trainData,trainLabels] = get_data_set('train');

% load('trainData.mat'); % 10000*1900
% load('yi.mat');        % 136*1900     

%% STEP 2: Train the first sparse autoencoder
%  This trains the first sparse autoencoder on the unlabelled training
%  images
%  Randomly initialize the parameters for layer 1 

addpath minFunc/
options.Method = 'lbfgs'; % optimization algorithm
options.maxIter = 400; % Maximum number of iterations of L-BFGS to run aslýnda 400
options.display = 'on';
%% Uncomment if you extract features meanwhile 
% sae1Theta = initializeParameters(hiddenSizeL1, inputSize);
t1 = tic;
% [sae1OptTheta, cost] = minFunc(@(p) sparseAutoencoderCost(...
%     p, inputSize, hiddenSizeL1, lambda, sparsityParam, beta, trainData), ...
%     sae1Theta, options);
load('sae1OptTheta.mat');
toc(t1);
% -------------------------------------------------------------------------
%%======================================================================
%% STEP 2: Train the second sparse autoencoder or your classifier  
%  This trains the second sparse autoencoder on the first autoencoder
%  featurs.
%% Uncomment if you are wxtracting sea1features in meanwhile
% [sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
%                                         inputSize, trainData);
% save sae1Features;
load('sae1Features.mat')

%% 2nd sparse auto encoder
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the second layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL2" and an inputsize of
%                "hiddenSizeL1"
%
%                You should store the optimal parameters in sae2OptTheta

t2 = tic;
% [sae2OptTheta, cost] = minFunc(@(p) sparseAutoencoderCost(...
%     p, hiddenSizeL1, hiddenSizeL2, lambda, sparsityParam, beta, sae1Features), ...
%     sae2Theta, options);
% toc(t2);
% save sae2OptTheta;
%  pause(5);
% [sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
%                                         hiddenSizeL1, sae1Features);
% save sae2Features;

load sae2Features;
%  pause(5);
%% STEP 3 : Train a multilayer perceptron
t3 = tic;
% perceptronTheta = initializePerceptronParams(outputSize,perceptronSize,hiddenSizeL2);
% perceptronOptions.Method = 'lbfgs';
% perceptronOptions.display = 'on';
% perceptronOptions.maxIter = 400; % aslýnda 400
% % perceptronun inpt size i hiddensizel1 , arada da perceptron size kadar
% % nöron var
% [saeMlpOptTheta, cost] = minFunc(@(p) mlpCost(...
%     p, hiddenSizeL2,perceptronSize,outputSize,lambda,sae2Features,yi), ...
%     perceptronTheta, perceptronOptions);
% % load('saeMlpOptTheta.mat');
% toc(t3);
% 
% save saeMlpOptTheta;
load saeMlpOptTheta;
 pause(5);
%% STEP 5: Finetune 

% Implement the stackedAECost to give the combined cost of the whole model
% then run this cell.

% Initialize the stack using the parameters learned
stack = cell(1,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);

stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeMlpOptTheta ; stackparams ];

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the deep network, hidden size here refers to the '
%                dimension of the input to the classifier, which corresponds 
%                to "hiddenSizeL2".
%
%
DeepOptions.Method = 'lbfgs';
DeepOptions.display = 'on';
DeepOptions.maxIter = 250; % aslýnda 100
t4 = tic;
[stackedAEOptTheta, cost] = minFunc( @(p) stackedAECost(p, ...
    inputSize, hiddenSizeL2, outputSize, netconfig, lambda, trainData, yi,perceptronSize), ...
    stackedAETheta, DeepOptions);
save stackedAEOptTheta;
% load stackedAEOptTheta
 pause(5);
toc(t4);
%% STEP 6: Test 
%  Instructions: You will need to complete the code in stackedAEPredict.m
%                before running this part of the code
%

% Get labelled test images
% Note that we apply the same kind of preprocessing as the training set
% [testData,testLabels] = get_multipie_data('test');
% testLabels(testLabels == 0) = 10; % Remap 0 to 10
load('testData.mat');
load('validation.mat');
load('denormalizedImages');
denormalizedImages=denormalizedImages(:,1901:end);

%% fine tune suz 
[pred1] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          outputSize, perceptronSize, netconfig, denormalizedImages);
save pred1 ;
 pause(5);
                      
cost_err= 0.5*sumsqr(pred1-validation);% J(w,b) cost
% acc = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', cost_err );
%% fine tunelu
[pred2] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          outputSize, perceptronSize, netconfig, denormalizedImages);
save pred2 ;
 pause(5);

% for i = 1: size(testData,2)
%     imshow(reshape(testData(:,i),100,100));
%     hold on;
%     plot(pred(1:68,i),pred(69:136,i),'r.','MarkerSize',20);
%     pause(5);
%     close all;
% end
cost_err= 0.5*sumsqr(pred2-validation);% J(w,b) cost
% acc = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', cost_err );

% Accuracy is the proportion of correctly classified images
% The results for our implementation were:
%
% Before Finetuning Test Accuracy: 87.7%
% After Finetuning Test Accuracy:  97.6%
%
% If your values are too low (accuracy less than 95%), you should check 
% your code for errors, and make sure you are training on the 
% entire data set of 60000 28x28 training images 
% (unless you modified the loading code, this should be the case)
