%%  Stacked Autoencoder implementation based on the courses of NG in CS294A/CS294W
% STEP 0: Parameter initialization

% About train data 
inputSize       = 50 * 50; % image size
outputSize      = 68*2+2;  % 136 coordinates + 2 pose classes 

% About network architecture
hiddenSizeL1    = 1600;    % first layer auto encoder extracts 1600 features 
hiddenSizeL2    = 900;     % second layer auto encoder extracts 900 features
hiddenSizeL3    = 400;     % third layer auto encoder extracts 400 features

% About network parameters 
sparsityParam   = 0.1;      % desired average activation of the hidden units.
                            % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
                            %  in the lecture notes). 
lambda          = 1e-2;     % weight decay parameter       
beta            = 3;        % weight of sparsity penalty term       


%% STEP 1: Load train data from three datasets: LFPW,AFW and HELEN
% Uncomment if you fetch data set in meanwhile 
% [train_images,yi,pose_labels] = get_raw_data_set('GCNZeroOneNormalization',0);
% normalized_pose =reshape(mat2gray(pose_labels(:)),size(pose_labels,1),size(pose_labels,2));
% 
% smap = vertcat(yi,normalized_pose');
 load('train_images.mat');
 load('smap.mat');

%% STEP 2.a: Layer 1 : Train the first sparse autoencoder 2500 * 1600

addpath minFunc/
options.Method = 'lbfgs'; % optimization algorithm
options.maxIter = 500;    % Maximum number of iterations of L-BFGS to run aslýnda 400
options.display = 'on';

sae1Theta = initializeParameters(hiddenSizeL1, inputSize);
t1 = tic;
[sae1OptTheta, cost] = minFunc(@(p) sparseAutoencoderCost(...
    p, inputSize, hiddenSizeL1, lambda, sparsityParam, beta, train_images), ...
    sae1Theta, options);
disp('auto encoder cost');
% save 'sae1OrptTheta.mat' sae1OptTheta;
load('sae1OptTheta.mat');
toc(t1);
% pause(4);
disp('Layer 1 : Theta calculated');


%% STEP 2.b: Feed Forward First Layer
% Uncomment if you are extracting sea1features in meanwhile
% [sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
%                                         inputSize, train_images);
% pause(4);
% save 'sae1Features.mat' sae1Features;
disp('Layer 1 : Features calculated');
load('sae1Features.mat');
% Normalize Features
sae1Features = reshape(mat2gray(sae1Features(:)),size(sae1Features,1),size(sae1Features,2));

%% STEP 3.a :  Layer 2 : Train the second sparse autoencoder : 1600*900

% sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);
% 
% t2 = tic;
% 
% [sae2OptTheta, cost] = minFunc(@(p) sparseAutoencoderCost(...
%     p, hiddenSizeL1, hiddenSizeL2, lambda, sparsityParam, beta, sae1Features), ...
%     sae2Theta, options);
% toc(t2);
% 
% pause(4);
% save 'sae2OptTheta.mat' sae2OptTheta;
load('sae2OptTheta.mat');
disp('Layer 2 : Thetas calculated');

%% STEP 3.b: Feed Forward Second Layer

% [sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
%                                         hiddenSizeL1, sae1Features);
% pause(4);
% save sae2Features;
load('sae2Features.mat');
% Normalize Features
sae2Features = reshape(mat2gray(sae2Features(:)),size(sae2Features,1),size(sae2Features,2));
disp('Layer 2 : Features calculated');

%% STEP 4.a :  Layer 3 : Train the third sparse autoencoder : 900*400
% sae3Theta = initializeParameters(hiddenSizeL3, hiddenSizeL2);
% t3 = tic;
% [sae3OptTheta, cost] = minFunc(@(p) sparseAutoencoderCost(...
%     p, hiddenSizeL2, hiddenSizeL3, lambda, sparsityParam, beta, sae2Features), ...
%     sae3Theta, options);
% toc(t3);
% 
% pause(4);
% save 'sae3OptTheta.mat' sae3OptTheta;
load('sae3OptTheta.mat');
disp('Layer 3 : Thetas calculated');

% load('sae3OptTheta.mat');
%% STEP 4.b: Feed Forward Third Layer

% [sae3Features] = feedForwardAutoencoder(sae3OptTheta, hiddenSizeL3, ...
%                                         hiddenSizeL2, sae2Features);
% save sae3Features;
load('sae3Features.mat');
disp('Layer 3 : Features calculated');
% pause(4);
% Normalize Features
sae3Features = reshape(mat2gray(sae3Features(:)),size(sae3Features,1),size(sae3Features,2));
disp('Layer 3 : Features calculated');

%% STEP 5 : Train a multilayer perceptron
t4 = tic;
% perceptronTheta = initializePerceptronParams(outputSize,hiddenSizeL3);
% perceptronOptions.Method = 'lbfgs';
% perceptronOptions.display = 'on';
% perceptronOptions.maxIter = 400; 
% [saeMlpOptTheta, cost] = minFunc(@(p) mlpCost(...
%     p, hiddenSizeL3,outputSize,lambda,sae3Features,smap), ...
%     perceptronTheta, perceptronOptions);
toc(t4);
% 
% save 'saeMlpOptTheta.mat' saeMlpOptTheta;

load('saeMlpOptTheta.mat');
disp('Perceptron has trained ');

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

stack{3}.w = reshape(sae3OptTheta(1:hiddenSizeL3*hiddenSizeL2), ...
                     hiddenSizeL3, hiddenSizeL2);
stack{3}.b = sae2OptTheta(2*hiddenSizeL3*hiddenSizeL2+1:2*hiddenSizeL3*hiddenSizeL2+hiddenSizeL3);
% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeMlpOptTheta ; stackparams ];

DeepOptions.Method = 'lbfgs';
DeepOptions.display = 'on';
DeepOptions.maxIter = 250; % aslýnda 100
t4 = tic;
% [stackedAEOptTheta, cost] = minFunc( @(p) stackedAECost(p, ...
%     inputSize, hiddenSizeL3, outputSize, netconfig, lambda, train_images, smap), ...
%     stackedAETheta, DeepOptions);
% save stackedAEOptTheta;
% pause(5);
load('stackedAEOptTheta.mat');
toc(t4);
disp('Ready to test');
%% STEP 6: Test : Load denormalized test images

% Get labelled test images
% Note that we apply the same kind of preprocessing as the training set

% [testImages,validation,test_pose_labels] = get_raw_test_set('normalized',0);
% save testImages.mat
% save validation.mat
% save test_pose_labels.mat
load('testImages.mat');
%% STEP 6: Test : Run Netwotk on Test Images
% [landmark_prd,pose_prd] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL3, ...
%                           outputSize, netconfig, testImages);
                      
                      [landmark_prd_on_train,pose_prd_on_train] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL3, ...
                          outputSize, netconfig, train_images);
                      
save landmark_prd_on_train ;
save pose_prd_on_train;
pause(5);

what_is_error



