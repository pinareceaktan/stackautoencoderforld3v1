% PIAN (Pose induced autoencoder) implementation
% Based on the paper

% Datasets :
% LFPW
% HELEN
% AFW
tic;

%% Fetch datasets
programroot = 'C:\Program Files\MATLAB\ml\SAN';
cd(programroot);
% lfpwdataset = strcat(programroot,'\lfw');
% datasetObj = dir(lfpwdataset);
% datasetCount = size(datasetObj,1);
%
% % counter = 1;
% % for i = 3:datasetCount % roll over the directories
% %     subjectname = datasetObj (i).name;
% %     subjectfolder = strcat(lfpwdataset,'\',subjectname);
% %     subjobj = dir(subjectfolder);
% %     subjectcount = size(subjobj,1);
% %     for j = 3:subjectcount % rolling in the subjects folder
% %         disp(counter);
% %         content(counter).name = {strcat(subjectfolder,'\',subjobj(j).name) };
% %         content(counter).subjname = {subjectname};
% %         counter = counter+1;
% %     end
% % end
%
% % load('content.mat');
% % unqsubjects = unique([content.subjname]);
% % %% 10 fold cross validation :
% % groupingVar = unqsubjects;
% % cvo = cvpartition(groupingVar,'k',10);
% % fold = 1; % 1..10
% % trainingind = find(training(cvo,fold)== 1);
% % % Testing set
% % testingind = find(test(cvo,fold)== 1);
%
% trainpath = 'C:\Program Files\MATLAB\ml\SAN\train';
% trainobj = dir(trainpath); % 811 imaj
% % cell for training objects
% for i = 3:size(trainobj,1)
%     im =  imread(char(strcat(trainpath,'\',trainobj(i).name)));
%     if size(im,3) == 3
%         im = rgb2gray(im);
%     end
%     tempim =  imresize(im, [255 255]);
%     ind = i-2;
%     traincells(ind) = {tempim};
% %     trainindex(ind,1) = {trainobj(i).name};
%     disp(['On training : ' num2str(ind) ] );
% end;
% cell for testing objects
% testpath = 'C:\Program Files\MATLAB\ml\SAN\test';
% testobj = dir(testpath); % 226 imaj
% testgt     = 'C:\Program Files\MATLAB\ml\SAN\testgt';
% testgtobj = dir(testgt);
% for i = 3:size(testobj,1)
%     im =  imread(char(strcat(testpath,'\',testobj(i).name)));
%     if size(im,3) == 3
%         im = rgb2gray(im);
%     end
%     tempim =  imresize(im, [255 255]);
%     ind = i-2;
%     testcells(ind) = {tempim};
%     disp(['On testing : ' num2str(ind) ] );
% end

load('traincells.mat');
load('trainindex.mat');

%% Run viola jones face detector
% css = 1;
% for i = 1: size(traincells,2)
%
%     detector = vision.CascadeObjectDetector;
%     bb = step(detector,traincells{1,i});
%     img = traincells{1,i};
%     traincellsbb(i) = {img(bb(1):(bb(1)+bb(3)),bb(2):(bb(2)+bb(4)))};
%     clearvars detector bb
% end
load('traincellsbb.mat');


%% Normalize images
% for i =1:size(traincellsbb,2)
%     img = im2double(traincellsbb{1,i});
%     temp = (img(:) - min(img(:))) / ( max(img(:)) - min(img(:)) );
%     temp2 =  imresize(temp, [50 50]);
%     normalizedtraincells(i) = {reshape(temp2, size(temp2, 1), size(temp2,2))};
%     disp(['On training : ' num2str(i) ] );
%     clearvars temp temp2
% end
load('normalizedtraincells.mat')
% for i = 1:size(testcells,2)
%     img = im2double(testcells{1,i});
%     temp = (img(:) - min(img(:))) / ( max(img(:)) - min(img(:)) );
%     normalizedtestcells(i) = {reshape(temp, size(img, 1), size(img,2))};
% 	disp(['On testing : ' num2str(i) ] );
%
% end
% load('normalizedtraincells.mat');

%% Fetch ground truth data for train objects
% traingt     = 'C:\Program Files\MATLAB\ml\SAN\traingt';
% traingtobj = dir(traingt);
%
% startRow = 4;
% endRow = 71;
% formatSpec = '%10f%f%[^\n\r]';
%
% for i = 3:size(traingtobj,1)
%     ind = i-2;
% % see if it the pts and png files are matches
% %     disp('---------');
% %     disp(trainindex(ind));
% %     disp(traingtobj(i).name);
%     filename = strcat(traingt,'\',traingtobj(i).name);
%     fileID = fopen(filename,'r');
%     dataArray = textscan(fileID, formatSpec, endRow-startRow+1, 'Delimiter', '', 'WhiteSpace', '', 'HeaderLines', startRow-1, 'ReturnOnError', false);
%     trainGroundTruth(ind) = {horzcat( dataArray{:,1} , dataArray{:,2})};
%     fclose(fileID);
% end
load('traingroundtruth.mat');
%% First Network :
for i = 1:size(normalizedtraincells,2)
    %     xi(:,i)=
    xi(:,i)=  (reshape(normalizedtraincells{1,i},numel(normalizedtraincells{1,i}),1));
    gi(:,i)= (reshape(trainGroundTruth{1,i},numel(trainGroundTruth{1,i}),1));
end

%% Network Architecture
n = [50 ; 100 ; 150 ; 300 ;500]; % dimension of theta vector of n hidden units
hiddenUnits1 = n(1);
hiddenUnits2 = n(1);

s1 = 50*50; % input layer
sn1 = hiddenUnits1; % hidden layer1
sn2 = hiddenUnits2; % hidden layer2
s3 = 50*50; % output layer

numclasses = 68;

network = [s1;sn1;s3];
nl = 3; % number of layers


%% Parameters Initialization :
% Random initialization :
init_epsilon  = sqrt(6) / sqrt(sn1+s1+1);
Theta(1)= {[rand((sn1),(s1))*2*(init_epsilon)-init_epsilon ]}; % number of units in hidden layer * number of units input layer
Theta(2) = {[rand((s3),sn1)*2*(init_epsilon)-init_epsilon]};  % number of units in output layer * number of units hidden layer
b1 = Theta{1,1}(:,1); % number of units in input layer*1
b2 = Theta{1,2}(:,1); % s1 = s3 its an autoencoder number of units in hidden layer *1
% Constant params initialization :
sc = 0.05; % sparsityParam
alfa = 0.01;
lambda = 0.0001;
beta = 3;
cost = 0;
W1grad = zeros(size(Theta{1,1}));
W2grad = zeros(size(Theta{1,2}));
b1grad = zeros(size(b1));
b2grad = zeros(size(b2));
%% Autoencoder :

m = size(xi,2);
%% Forward propogation
bias = b1;
a(:,1) = {xi}; % input
for i = 2:3 %loop through the hidden layer
    l = i ; % next layer
    lpre = i-1; % previous layer
    z(l) =  {Theta{1,lpre}*a{1,lpre}+repmat(bias,1,m)};
    a(l)=    {sigmoid(z{1,l})};
    bias = b2 ;
end
rho = sum(a{1,2})/m; % mean of the activations in hidden layer
cost_err = 1/2*norm(a{1,3}-a{1,1}); % J(w,b) cost

%% Backpropagation : Error of node j in layer l

for i = nl:-1:2 % for each layer back to the front
    lb = i;
    if lb == nl % is it the output layer 3.layer
        delta(1,lb)= {(a{1,3}-a{1,1}).*sigmoidinv(z{1,lb})} ;
    else
        % hidden layer              
        delta(1,lb) = {(Theta{1,lb}'*delta{1,lb+1}+beta*repmat((-(sc./rho)+(1-sc)./(1-rho)),sn1,1)).*sigmoidinv(z{1,lb})};
    end
    % Compute the derivatives
    partialw(lb) = {delta{1,lb}*a{lb-1}.'};
    partialb(lb-1) = {delta{1,lb}};
end
%% Gradient descent to decrease the cost function

W1grad = partialw{1,2}*1/m+lambda*Theta{1,1};
b1grad =  1/m*partialb{1,1};
W2grad = partialw{1,3}*1/m+lambda*Theta{1,2};
b2grad = 1/m*partialb{1,2};


%% Cost
KLdiv = sc*log(sc./rho) +(1 - sc)*log((1 - sc)./(1 - rho));

cost_err = cost_err/m;
cost_weights = lambda/2*(sum(Theta{1,1}(:).^2) + sum(Theta{1,2}(:).^2)); % w regularization
cost_sparse = beta*sum(KLdiv); % induce "sparsity"
% returns 
cost = cost_err + cost_weights + cost_sparse;
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];


%% Optimization on first stack 
cd('C:\Program Files\MATLAB\ml\SAN\minFunc');

options.Method = 'lbfgs'; % optimization algorithm
options.maxIter = 400; % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
sae1Theta  = [Theta{1,1}(:) ; Theta{1,2}(:) ; b1(:) ; b2(:)];

t1 = tic;
[sae1OptTheta, cost] = minFunc([cost,grad],sae1Theta, options);
toc(t1);



% Initializaa cost function
% @J(theta)= y* log(theta(x)+ (1-y)*log(1-theta(x)));

% Theta optimization : gradient checking
% for i = 1:size(ThetaVec,1)
%     thetaPlus = theta;
%     thetaPlus(i) = thetaPlus(i)+ epsilon;
%     thetaMinus = theta;
%     thetaMinus(i) = thetaMinus(i) - epsilon;
%     gradApprox(i) = (J(thetaPlus)-J(thetaMinus))/(2*epsilon);
% end




toc;



