%% Multi layer perceptron 
%
%% STEP 0:  Load data set
% [trainData,trainLabels] = get_multipie_data('train');
% load('trainDataall.mat');
% load('groundTruth.mat');
% trainData = trainDataAll(:,1:1900);
% testData  = trainDataAll(:,1901:end);
% yi = groundTruth(:,1:1900);
% validation = groundTruth(:,1901:end);
load('trainData.mat')
load('testData.mat')
load('yi.mat')
load('validation.mat')
load('denormalizedImages.mat')
load('normalizedLabels.mat', 'normalizedLabels')

%% STEP 1 : Parameter Initialization 
InputSize       =  100*100;
OutputSize    =  68*2;
hiddenSizeL1 =  200;
lambda = 3e-3; 

%% STEP 2 : Train a multilayer perceptron
perceptronTheta = initializePerceptronParams(OutputSize,hiddenSizeL1,InputSize);

%% STEP 3 : Optimization Algorithm 
addpath minFunc/
options.Method = 'lbfgs'; % optimization algorithm
options.maxIter = 400; % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
[MlpOptTheta, cost] = minFunc(@(p) mlpCost(...
    p, InputSize, hiddenSizeL1,OutputSize, lambda, trainData,yi), ...
    perceptronTheta, options);
denormalizedImages=denormalizedImages(:,1:1900);

[landmarks] = feedforwardperceptron(MlpOptTheta,hiddenSizeL1,...
                        InputSize,OutputSize,denormalizedImages);
imshow(reshape(trainData(:,1),100,100));
hold on;
plot(landmarks(1:68),landmarks(69:136),'r.','MarkerSize',20);

