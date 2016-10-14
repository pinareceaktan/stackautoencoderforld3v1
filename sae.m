%%  Stacked Autoencoder implementation based on the courses of NG in CS294A/CS294W
%% STEP 0: Parameter initialization
inputSize       = 50 * 50;% image size
outputSize      = 68*2;     % 136 coordinates will be extracted
hiddenSizeL1    = 1600;     % first layer auto encoder extracts 1600 features 
hiddenSizeL2    = 900;      % second layer auto encoder extracts 900 features
hiddenSizeL3    = 400;      % third layer auto encoder extracts 400 features
perceptronSize  = 200;      % perceptron has 200 neurons
 
sparsityParam   = 0.1;      % desired average activation of the hidden units.
                            % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
                            %  in the lecture notes). 
lambda          = 1e-2;     % weight decay parameter       
beta            = 3;        % weight of sparsity penalty term       


%% STEP 1: Load train data from three datasets: LFPW,AFW and HELEN
%% Uncomment if you fetch data set in meanwhile 
[train_images,yi,pose_labels] = get_raw_data_set('GCNMean0Std1Normalized');
% load('train_images.mat'); % 2500*4573
% load('yi.mat');           % 2500*4573     
disp('eben')
%% STEP 2: Layer 1 : Train the first sparse autoencoder
%  This trains the first sparse autoencoder on the unlabelled training
%  images
%  Randomly initialize the parameters for layer 1 

addpath minFunc/
options.Method = 'lbfgs'; % optimization algorithm
options.maxIter = 500; % Maximum number of iterations of L-BFGS to run aslýnda 400
options.display = 'on';
%% Uncomment if you extract features meanwhile 
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);
t1 = tic;
[sae1OptTheta, cost] = minFunc(@(p) sparseAutoencoderCost(...
    p, inputSize, hiddenSizeL1, lambda, sparsityParam, beta, train_images), ...
    sae1Theta, options);
disp('Layer 1 : Thetas calculated');
% load('sae1OptTheta.mat');
toc(t1);
% -------------------------------------------------------------------------
%%======================================================================
%% STEP 2: Feed Forward First Layer
%  This trains the second sparse autoencoder on the first autoencoder
%  featurs.
%% Uncomment if you are extracting sea1features in meanwhile
[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, train_images);
disp('Layer 1 : Features calculated');

% save sae1Features;
% load('sae1Features.mat')

%% STEP 3 :  Layer 2 : Train the second sparse autoencoder
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);
t2 = tic;
[sae2OptTheta, cost] = minFunc(@(p) sparseAutoencoderCost(...
    p, hiddenSizeL1, hiddenSizeL2, lambda, sparsityParam, beta, sae1Features), ...
    sae2Theta, options);
toc(t2);
disp('Layer 2 : Thetas calculated');
% save sae2OptTheta;
%% STEP 3: Feed Forward Second Layer

[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);
save sae2Features;
disp('Layer 2 is done');
load sae2Features;
%  pause(5);
%% STEP 4 : Train a multilayer perceptron
t3 = tic;
perceptronTheta = initializePerceptronParams(outputSize,hiddenSizeL3);
perceptronOptions.Method = 'lbfgs';
perceptronOptions.display = 'on';
perceptronOptions.maxIter = 400; 
[saeMlpOptTheta, cost] = minFunc(@(p) mlpCost(...
    p, hiddenSizeL2,perceptronSize,outputSize,lambda,sae2Features,yi), ...
    perceptronTheta, perceptronOptions);
% load('saeMlpOptTheta.mat');
toc(t3);
disp('Perceptron has trained ');
% save saeMlpOptTheta;
load saeMlpOptTheta;
 pause(5);
%% STEP 5: Finetune 

% Implement the stackedAECost to give the combined cost of the whole model
% then run this cell.

% % Initialize the stack using the parameters learned
% stack = cell(1,1);
% stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
%                      hiddenSizeL1, inputSize);
% stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
% 
% stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
%                      hiddenSizeL2, hiddenSizeL1);
% stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);
% 
% % Initialize the parameters for the deep model
% [stackparams, netconfig] = stack2params(stack);
% stackedAETheta = [ saeMlpOptTheta ; stackparams ];

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the deep network, hidden size here refers to the '
%                dimension of the input to the classifier, which corresponds 
%                to "hiddenSizeL2".
%
%
% DeepOptions.Method = 'lbfgs';
% DeepOptions.display = 'on';
% DeepOptions.maxIter = 250; % aslýnda 100
% t4 = tic;
% [stackedAEOptTheta, cost] = minFunc( @(p) stackedAECost(p, ...
%     inputSize, hiddenSizeL2, outputSize, netconfig, lambda, train_images, yi,perceptronSize), ...
%     stackedAETheta, DeepOptions);
% save stackedAEOptTheta;
load stackedAEOptTheta
%  pause(5);
toc(t4);
%% STEP 6: Test 
%  Instructions: You will need to complete the code in stackedAEPredict.m
%                before running this part of the code
%

% Get labelled test images
% Note that we apply the same kind of preprocessing as the training set
[testImages,validation,test_pose_labels] = get_raw_test_set();
load('test_data');
%% fine tune suz 
% [pred1] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
%                           outputSize, perceptronSize, netconfig, testImages);
% save pred1 ;
% pause(5);
             
load('pred1');
cost_err= 0.5*sumsqr(pred1-validation);% J(w,b) cost
% acc = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', cost_err );
%% fine tunelu
[pred2] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          outputSize, perceptronSize, netconfig, testImages);
save pred2 ;
pause(5);
load('pred2.mat');
load('denormalizedTestData.mat')
% for i = 1: size(testImages,2)
%     fig = figure;
%     subplot(1,2,1);
%     imshow(reshape(denormalizedTestData(i).face,100,100));
%     hold on;
%     plot(pred2(1:68,i),pred2(69:136,i),'r.','MarkerSize',20);
%     title('My Predictions')
%     subplot(1,2,2);
%     imshow(reshape(denormalizedTestData(i).face,100,100));
%     hold on;
%     plot(validation(1:68,i),validation(69:136,i),'r.','MarkerSize',20);
%     title('Ground Truth')
%     pause(5);
%     close all;
% end
% cost_err= 0.5*sumsqr(pred2-validation);% J(w,b) cost
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
