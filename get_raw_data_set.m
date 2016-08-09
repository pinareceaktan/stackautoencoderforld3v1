% function [images,labels] = get_data_set(dataset)
% function get_data_set_raw fetches data to prepare train and test sets
% for neural network implementation
% 3 datasets is in use which explanied below
% the datasets are from i-bug. 
%% Train Dataset 
% Three databases: LFPW [21],Helen [22] and AFW [23]. 
% 811 images from LFPW,
% 2000 images from Helen
% 337 images from AFW as
% Note : To further augment the data, we conduct translation
% and rotation to each image.
%% Test Dataset 
%   300k - 600 images (300 indoor 300 outdoor)
%
%% Zhu-Ramanan Face Detector 
% X. Zhu, D. Ramanan. "Face detection, pose estimation and landmark localization in the wild" 
% Computer Vision and Pattern Recognition (CVPR) Providence, Rhode Island, June 2012. 
% website : https://www.ics.uci.edu/~xzhu/face/
imsize = 100*100;
classsize = 68*2;
programRoot= pwd; 
%% LFPW Dataset :

% lfpwDataSetRoot     = strcat(programRoot,'\','datasetstuff','\LFPW');
% lfpwTrainFolder     = strcat(lfpwDataSetRoot,'\trainset');
% lfpwAnnotations     = strcat(lfpwDataSetRoot,'\annotations');
% dirContent          = dir(lfpwAnnotations);
% lfpwtrainImList     = cell(size(dirContent,1)-2,1);
% lfpwAnnotationList  = cell(size(dirContent,1)-2,1);
% 
% for i = 3 : size(dirContent,1)
%     lfpwtrainImList(i-2,1)     =  strcat(regexpi(dirContent(i).name,'\S*(?=\.pts)','match'),'.png');
%     lfpwAnnotationList(i-2,1)  =  {dirContent(i).name};
% end
% %% Create ground truth matrix for LFPW
% 
% % for i = 1:size(lfpwAnnotationList,1)
% %     try
% %       fileID                = fopen(strcat(lfpwAnnotations,'\',lfpwAnnotationList{i,1}));
% %       scannedText           = textscan(fileID,'%s');
% %       lfpwGroundTruth(i,1)  = lfpwtrainImList(i,1); 
% % %     stupid extension because gts are like ass 
% %       dim1 = scannedText{1,1}(6:2:140);
% %       dim2 = scannedText{1,1}(7:2:141);
% %     for j = 1 : size(dim1,1)
% %         dim1arr(j,1)= str2num(dim1{j,1});
% %         dim2arr(j,1)= str2num(dim2{j,1});
% %     end
% %     
% %         lfpwGroundTruth(i,2) = {[dim1arr dim2arr]};
% % 
% %     fclose(fileID);
% %     clear fileID scannedText dim1 dim2
% %     catch ME
% %            disp(num2str(i));
% %         disp(ME.identifier);
% %     continue;
% %     end
% % 
% % end
% % disp('sdfsdf');
% load('lfpwGroundTruth.mat');
% 
% for i = 1: size(lfpwtrainImList,1)
%    try
%     imagePath       = strcat(lfpwTrainFolder,'\',lfpwtrainImList{i,1});
%     gt_index        = find(strcmp(lfpwGroundTruth(:,1),lfpwtrainImList{i,1}));
%     image           = imread(imagePath);
%     disp(['1 Image: ' num2str(i) ' has loaded']);
%     groundTruth     = lfpwGroundTruth{gt_index,2};    
%     fixedImage      = image;
%     fixedgt         = groundTruth;
%     disp(['1.b Ground truth: ' num2str(i) ' has loaded']);
% 
% %% 2 a) Crop only the face frame using ground truth data  
%    disp('2..Cropping only the face out of the image')
%     % specifiy borders :
%     lefters         = [1:6];
%     righters        = [13:17];
%     deepers         = [7:12];
%     leftUppers      = [18:22];
%     rightUppers     = [23:27];
%     
%     rightIndes    = find( fixedgt(:,1)==max(fixedgt(righters,1)));
%     leftIndes     = find( fixedgt(:,1)==min(fixedgt(lefters,1)));
%     leftTopIndes  = find( fixedgt(:,2)==min(fixedgt(leftUppers,2)));
%     rightTopIndes = find( fixedgt(:,2)==min(fixedgt(rightUppers,2)));
%     downIndes     = find( fixedgt(:,2)==max(fixedgt(deepers,2)));
%     
%     mostRightInd   = rightIndes(1,1);
%     mostLeftInd    = leftIndes(1,1);
%     mostLeftTopInd = leftTopIndes(1,1);
%     mostRightTopInd= rightTopIndes(1,1);
%     mostDownInd    = downIndes(1,1);
%     
%     rightBound   = [fixedgt(mostRightInd,1),fixedgt(mostRightInd,2)];
%     leftBound    = [fixedgt(mostLeftInd,1),fixedgt(mostLeftInd,2)];
%     upperBound   = [(fixedgt(mostLeftTopInd,1)+fixedgt(mostRightTopInd,1)),(fixedgt(mostLeftTopInd,2)+fixedgt(mostRightTopInd,2))]/2;
%     bottomBound  = [fixedgt(mostDownInd,1),fixedgt(mostDownInd,2)];
%     
%     % shift'em a little
%     rightBound(1,1) = rightBound(1,1)+20;
%     leftBound(1,1)  = leftBound(1,1)-20;
%     upperBound(1,2) = upperBound(1,2)-20;
%     bottomBound(1,2)= bottomBound(1,2)+20;
%     % check borders
%     if leftBound(1,1)<0  % left border
%         leftBound(1,1) = 0;
%     end
%     if rightBound(1,1)>size(fixedImage,2) % right border
%         rightBound(1,1) = size(fixedImage,2);
%     end
%     if upperBound(1,2)<0
%         upperBound(1,2)= 0;
%     end
%     if   bottomBound(1,2) > size(fixedImage,1)
%         bottomBound(1,2) = size(fixedImage,1);
%     end
%     gbbox= [leftBound(1,1),upperBound(1,2), rightBound(1,1)-leftBound(1,1),bottomBound(1,2)-upperBound(1,2)];
%     gface  =  fixedImage(gbbox(2):(gbbox(2)+gbbox(4)),gbbox(1):(gbbox(1)+gbbox(3)));
%     disp('2.a Face has extracted from image')
% %% 2 b) Shift ground truths accordingly  
%     gshift = double(gbbox(1:2));
%     fixedgt =  (horzcat((fixedgt(:,1)-gshift(1)),(fixedgt(:,2)-gshift(2))));
% %% 3 ) Resize
%     nonresizedsize  = size(gface);
%     gface           = imresize(gface,[100 100]);
%     resizedsize     = size(gface);
%     scalex          = nonresizedsize(1)/resizedsize(1);
%     scaley          = nonresizedsize(1,2)/resizedsize(1,2);
%     fixedgt         = horzcat(fixedgt(:,1)/scaley,fixedgt(:,2)/scalex);
%     
%     disp('3 Resizing the face has done')
%  %% 4.1 ) Draw and save results
%     h = figure ;
%     imshow(gface);
%     hold on
%     plot(fixedgt(:,1),fixedgt(:,2),'r.','MarkerSize',10)
%     title(['image : ' num2str(i)])
%     
%     saveas(h,char(strcat('resultsLFPW/',lfpwtrainImList{i,1})))
%     disp('4.1 Face and gts has saved visually')
% %% 4.2 ) Normalization : Scale intensity values
%         gface   =  im2double(gface);
%         gface   = normalizePic(gface);
%         fixedgt = normalizePic(fixedgt);
%         disp('4.2 scaling intensity values has done')
% %% 4.3 ) Final : Save data mat
%         lfpwdata(i).face        =  reshape(gface,imsize,1);
%         lfpwdata(i).groundtruth =  reshape(fixedgt,classsize,1);
%           
%         disp(['4.3 ' num2str(i) ' saved to dataset']);
%        
% 
%   
%     clear imagePath image gface  landmarkPoints fixedgt gbbox fixedImage
%     clear rightBound rightBound upperBound bottomBound
%     close all;
% 
%   pause(1)
%    catch ME
%        fileID = fopen('logfile_lfpw.txt','a');
%        fprintf(fileID,'%20s %40s %3d\n',char(lfpwtrainImList{i,1}),(ME.identifier),(ME.stack.line));
%        fclose(fileID);
%        continue;
%    end
% end

%% AFW DATASET
% REQUIRES WORKING ZHU-RAMANAN  
%% HELEN DATASET
helenDataSetRoot = strcat(programRoot,'\','datasetstuff\HELEN68'); 
helenTrainFolder = strcat(helenDataSetRoot,'\trainset');
helenAnnotations = strcat(helenDataSetRoot,'\annotations');
helentrainImList = strcat(helenDataSetRoot,'\train_im_list.txt');
fileID           = fopen(helentrainImList);
scannedText = textscan(fileID,'%s');
trainImages = scannedText{1,1};
%% Create ground truth matrix for Helen

% dirContent  =   dir(helenAnnotations);
% for i = 3:size(dirContent,1)
%     try
%     counter = i-2;
%     fileID = fopen(strcat(helenAnnotations,'\',dirContent(i).name));
%     scannedText = textscan(fileID,'%s');
%     helenGroundTruth(counter,1) = (strcat(regexpi(dirContent(i).name,'\S*(?=\.pts)','match'),'.jpg')); 
%     stupid extension because gts are like ass 
%     dim1 = scannedText{1,1}(6:2:140);
%     dim2 = scannedText{1,1}(7:2:141);
%     for j = 1 : size(dim1,1)
%         dim1arr(j,1)= str2num(dim1{j,1});
%         dim2arr(j,1)= str2num(dim2{j,1});
%     end
% %     helenGroundTruth(counter,2) = {[scannedText{1,1}(6:2:140) scannedText{1,1}(7:2:141)]};
%     helenGroundTruth(counter,2) = {[dim1arr dim2arr]};
% 
%     fclose(fileID);
%     clear fileID scannedText counter dim1 dim2 
%     catch ME
%         disp(num2str(i));
%         disp(ME.identifier);
%     continue;
%     end
% end
% fclose all
% clear dirContent
% run_compilers; % compile zhu-ramanan face detector
load('helenGroundTruthRaw.mat') % load ground-truth matrix

for i = 1: size(trainImages,1)
   try
    file_num        = regexpi(trainImages{i,1},'\d*(?=\_)','match');
    subject_num     = regexpi(trainImages{i,1},'(?<=_)\d*','match'); 
    imagePath       = strcat(helenTrainFolder,'\',trainImages{i,1},'.jpg');
    gt_index        = find(strcmp(helenGroundTruth(:,1),strcat(file_num,'_',subject_num,'.jpg')));
    image           = imread(imagePath);
    disp(['1 Image: ' num2str(i) ' has loaded']);
    groundTruth     = helenGroundTruth{gt_index,2};    
    fixedImage      = image;
    fixedgt         = groundTruth;
    disp(['1.b Ground truth: ' num2str(i) ' has loaded']);

%% 2 a) Crop only the face frame using ground truth data  
   disp('2..Cropping only the face out of the image')
    % specifiy borders :
    lefters         = [1:6];
    righters        = [13:17];
    deepers         = [7:12];
    leftUppers      = [18:22];
    rightUppers     = [23:27];
    
    rightIndes    = find( fixedgt(:,1)==max(fixedgt(righters,1)));
    leftIndes     = find( fixedgt(:,1)==min(fixedgt(lefters,1)));
    leftTopIndes  = find( fixedgt(:,2)==min(fixedgt(leftUppers,2)));
    rightTopIndes = find( fixedgt(:,2)==min(fixedgt(rightUppers,2)));
    downIndes     = find( fixedgt(:,2)==max(fixedgt(deepers,2)));
    
    mostRightInd   = rightIndes(1,1);
    mostLeftInd    = leftIndes(1,1);
    mostLeftTopInd = leftTopIndes(1,1);
    mostRightTopInd= rightTopIndes(1,1);
    mostDownInd    = downIndes(1,1);
    
    rightBound   = [fixedgt(mostRightInd,1),fixedgt(mostRightInd,2)];
    leftBound    = [fixedgt(mostLeftInd,1),fixedgt(mostLeftInd,2)];
    upperBound   = [(fixedgt(mostLeftTopInd,1)+fixedgt(mostRightTopInd,1)),(fixedgt(mostLeftTopInd,2)+fixedgt(mostRightTopInd,2))]/2;
    bottomBound  = [fixedgt(mostDownInd,1),fixedgt(mostDownInd,2)];
    % shift'em a little
    rightBound(1,1) = rightBound(1,1)+20;
    leftBound(1,1)  = leftBound(1,1)-20;
    upperBound(1,2) = upperBound(1,2)-20;
    bottomBound(1,2)= bottomBound(1,2)+20;
    % check borders
    if leftBound(1,1)<0  % left border
        leftBound(1,1) = 0;
    end
    if rightBound(1,1)>size(fixedImage,2) % right border
        rightBound(1,1) = size(fixedImage,2);
    end
    if upperBound(1,2)<0
        upperBound(1,2)= 0;
    end
    if   bottomBound(1,2) > size(fixedImage,1)
        bottomBound(1,2) = size(fixedImage,1);
    end
    gbbox= [leftBound(1,1),upperBound(1,2), rightBound(1,1)-leftBound(1,1),bottomBound(1,2)-upperBound(1,2)];
    gface  =  fixedImage(gbbox(2):(gbbox(2)+gbbox(4)),gbbox(1):(gbbox(1)+gbbox(3)));
    disp('2.a Face has extracted from image')
%% 2 b) Shift ground truths accordingly  
    gshift = double(gbbox(1:2));
    fixedgt =  (horzcat((fixedgt(:,1)-gshift(1)),(fixedgt(:,2)-gshift(2))));
%% 3 ) Resize
    nonresizedsize  = size(gface);
    gface            = imresize(gface,[100 100]);
    resizedsize     = size(gface);
    scalex          = nonresizedsize(1)/resizedsize(1);
    scaley          = nonresizedsize(1,2)/resizedsize(1,2);
    fixedgt         = horzcat(fixedgt(:,1)/scaley,fixedgt(:,2)/scalex);
    
    disp('3 Resizing the face has done')
%% 4.1 ) Draw and save results
    h = figure ;
    imshow(gface);
    hold on
    plot(fixedgt(:,1),fixedgt(:,2),'r.','MarkerSize',10)
    title(['image : ' num2str(i)])
    
    saveas(h,char(strcat('resultsHELEN/',file_num,'_',subject_num, '.jpg')))
    disp('4.1 Face and gts has saved visually')
  
%% 4.2 ) Normalization : Scale intensity values
    gface   =  im2double(gface);
    gface   = normalizePic(gface);
    fixedgt = normalizePic(fixedgt);
    disp('4.2 scaling intensity values has done')
%% 4.3 ) Final : Save data mat
        helendata(i).face        =  reshape(gface,imsize,1);
        helendata(i).groundtruth =  reshape(fixedgt,classsize,1);
          
        disp(['4.3 ' num2str(i) ' saved to dataset']);
       

  
    clear imagePath image gface  landmarkPoints fixedgt gbbox fixedImage
    clear rightBound rightBound upperBound bottomBound
    close all;
    
  pause(1)
   catch ME
       fileID = fopen('logfile_v2.txt','a');
       fprintf(fileID,'%20s %40s %3d\n',char(strcat(file_num,'_',subject_num,'.jpg')),(ME.identifier),(ME.stack.line));
       fclose(fileID);
       continue;
   end
end

% DRAW
%     imshow(dataset(i).images);
%     hold on
%     plot(dataset(i).groundtruth(:,1),dataset(i).groundtruth(:,2),'r.','MarkerSize',20);

% labelsfile = 'L:\Labels\labels\051';
% addpath(labelsfile)
% dirinfo = dir(labelsfile);
% dirinfo = dirinfo(3:end);
% if nargin <1
%     dataset='train';
% espnd
% switch dataset
%     case 'train'
%      dirinfo = dirinfo(1:2200);
%      normalize = 1;
%     case 'test'
%      dirinfo = dirinfo(2201:end);
%      normalize = 0;
% end
% data = struct('images',[],'groundtruth',[]);
% 
% counter = 1;
% for i=1: numel(dirinfo)
%     fileparse = regexpi(dirinfo(i).name,'\d*(?=\_)','match'); % fetch ground truth
%     load(dirinfo(i).name); % load pts var
%     if numel(pts) ~= 136
%         disp([num2str(i) ' been removed from dataset']);
%     continue;
%     end
%     subject = fileparse{1,1};
%     session = fileparse{1,2};
%     record  = fileparse{1,3};
%     camera = strcat(fileparse{1,4}(1:2),'_',fileparse{1,4}(3));
%     illumination = fileparse{1,5};
%     fname = strcat(subject,'_',session,'_',record,'_',fileparse{1,4},'_',illumination,'.png');
%     addpath (strcat('L:\Multi-Pie\data\','session',session,'\multiview\',subject,'\',record,'\',camera));
%     im = imread(fname); % fetch image of given ground truth
%     disp('fetch image of given ground truth');
% %% Draw landmarks
% %     imshow(im);
% %     hold on
% %     plot(pts(:,1),pts(:,2),'r.','MarkerSize',20);
% %% Normalization : 
% % Step 1 : rgb2gray
% if size(im,3) == 3 
%     im = rgb2gray(im);
% end
% % Step 2 : Viola Jones 
% detector = vision.CascadeObjectDetector ;
% bb = step(detector,im);
% if size(bb,1) == 0
%     disp([num2str(i) ' been removed from dataset']);
%     continue;
% end
% [originx,originy] = find(bb(:,3:4)>150);
% bbop = bb(originx(1),:);
% face =  im(bbop(2):(bbop(2)+bbop(4)),bbop(1):(bbop(1)+bbop(3)));
% shift = bbop(1:2);
% % Step 3 : Resize
% nonresizedsize = size(face);
% face = imresize(face,[100 100]);
% resizedsize = size(face);
% scale = nonresizedsize(1)/resizedsize(1);
% % Step 4 : Scale intensity values
% if normalize
%     face =  im2double(face);
%     face = normalizePic(face);
%     disp('normalization done');
% end
% 
% data(counter).images = face;
%  %% ground truthlari çek ve normalize et
%  data(counter).groundtruth =  normalizePic(im2double(horzcat((pts(:,1)-shift(1))/scale,(pts(:,2)-shift(2))/scale)));
%  disp([num2str(i) ' saved to dataset']);
% %% Draw normalized landmarks
% %     imshow(dataset(i).images);
% %     hold on
% %     plot(dataset(i).groundtruth(:,1),dataset(i).groundtruth(:,2),'r.','MarkerSize',20);
%  clearvars detector bb face nonresizedsize resizedsize scale shift
%  counter = counter +1;
% end
% imsize = 100*100;
% classsize = 68*2;
% images = zeros(imsize,size(data,2));
% labels   = zeros(classsize,size(data,2));
% save dataset;
% for i = 1: size(data,2)
% %     images(:,i) = reshape(dataset(i).images,imsize,1);
%     labels(:,i) = reshape(data(i).groundtruth,classsize,1);
% end
cd(programRoot);