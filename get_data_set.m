% function [images,labels] = get_data_set(dataset)
% if normalize is 1 return normalize version of the data set 
% it is usually when you train your network
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

%% LFPW Dataset :
programRoot= pwd; 
% 
% % put csv file into a premature array : dataArray
% lfpwdataset = 'D:\DATASETS\LFPW- Labeled Face Parts in the Wild\kbvt_lfpw_v1_test.csv';
% delimiter = '\t';
% 
% formatSpec = '%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%[^\n\r]';
% fileID = fopen(lfpwdataset,'r');
% % raw form of the data  
% dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false); 
% fclose(fileID);
% addpath 'datasetstuff'
% load('dataArray.mat')
% Create images from urls : 
% for i = 1 : 1%size(dataArray,2) % loop through the fields 108
%     for j = 2 :  size(dataArray{i},1)
%        
% % outfilename = websave(filename,url)
% try
%     outfilename = websave(['dataset/lfpw_' num2str(j) '.jpg'],char(cellstr(dataArray{1,i}(j,1))));
%     disp(['item ' num2str(j) ':' outfilename ' was saved!'])
% 
% catch ME
%         disp(['item ' num2str(j) ' exites with --> ' ME.identifier]);
%         stat = which(['dataset/lfpw_' num2str(j) '.jpg.html']);
%         if ~strcmp(stat, '') 
%             delete(['dataset/lfpw_' num2str(j) '.jpg.html']);
%         end
%         clear stat
% end
%     end
% end
% 
% disp('downloading LFPW images and landmark points')
% 
% datasetRoot = 'C:\Users\FERA_ECE\Documents\MATLAB\stackautoencoderforld3\lfpwDataset';
% dirContent = dir(datasetRoot);
% lfpwimages = cell(size(dirContent(3:end),1),size(dataArray,2)-1); % -1 cause the workers wont be placed in array
% 
% for i = 3: size(dirContent,1) % loop through the images 
%     impath =  dirContent(i).name;
%     imnum = regexp(impath,'\d*','match');
%     imnum = str2double(imnum{1,1});
%   
%  
%     lfpwimages{i-2,1} =  imread(impath); % image
%     for j = 3 : size(dataArray,2) % loop through the fields starting with left_eyebrow_out_x
%         lfpwimages{i-2,j-1} = str2double(dataArray{1,j}(imnum)); % j-1 equals to 2 , and field 1 is already filled by the image array
%     end
%     
% 
%     clear impath imnum
% end
% addpath 'datasetstuff'
% load('lfpwimages.mat', 'lfpwimages');

% %% Add file names into the lpfwimages.dat
% lpfwroot = 'C:\Program Files\MATLAB\ml\stackautoencoderforld3v1\stackautoencoderforld3v1\datasetstuff\LPFW';
% lpfwcontent = dir(lpfwroot);
% for i = 3: size(lpfwcontent)
%     lfpwimages{i-2,107} = char(lpfwcontent(i).name);
% end
%  
clear delimiter fileID dataArray datasetRoot dirContent

%% AFW DATASET
% REQUIRES WORKING ZHU-RAMANAN  
%% HELEN DATASET
helenDataSetRoot = strcat(programRoot,'\','datasetstuff\HELEN'); 
helenTrainFolder = strcat(helenDataSetRoot,'\helen_train_dataset');
helenAnnotations = strcat(helenDataSetRoot,'\annotation');
trainImList      = strcat(helenDataSetRoot,'\train_im_list.txt');
fileID           = fopen(trainImList);
scannedText = textscan(fileID,'%s');
trainImages = scannedText{1,1};
%% Create ground truth matrix for Helen
clear fileID scannedText
dirContent  =   dir(helenAnnotations);
load('helenGroundTruth.mat')
% for i = 3:size(dirContent,1)
%     counter = i-2;
%     disp(counter);
%     fileID = fopen(strcat(helenAnnotations,'\',dirContent(i).name));
%     scannedText = textscan(fileID,'%d %c %d ');
%     helenGroundTruth(counter,1) = {(strcat(num2str(scannedText{1,1}(1,1)),'_',num2str(scannedText{1,3}(1,1)),'.jpg'))}; 
%     helenGroundTruth(counter,2) = {[scannedText{1,1}(2:end) scannedText{1,3}(2:end)]};
%     fclose(fileID);
%     clear fileID scannedText
% end
% fclose all
% clear dirContent
% run_compilers;
load('helenGroundTruth.mat'); % load ground-truth matrix

allGroundTruths = cell(size(trainImages,1),3);

for i =850: size(trainImages,1)
   try
    file_num        = regexpi(trainImages{i,1},'\d*(?=\_)','match');
    subject_num     = regexpi(trainImages{i,1},'(?<=_)\d*','match'); 
    imagePath       = strcat(helenTrainFolder,'\',trainImages{i,1},'.jpg');
    gt_index        = find(strcmp(helenGroundTruth(:,1),strcat(file_num,'_',subject_num,'.jpg')));
    image           = imread(imagePath);
    disp(['1..Image: ' num2str(i) ' has loaded']);
    groundTruth     = helenGroundTruth{gt_index,2};
    %% 0.1 ) ReScale Images and Ground Truths
    if size(image,1)>2000 || size(image,2)>2000
        imscale = 0.3;
    else
        imscale = 1;
    end
    fixedImage      = imresize(image,imscale);
    nonfixedsize    = size(image);  
    fixedsize       = size(fixedImage);
    scale           = nonfixedsize(1)/fixedsize(1);
    fixedgt         = horzcat(groundTruth(:,1)/scale,groundTruth(:,2)/scale);
    disp('2.. Image has scaled 0.3 smaller')
    %% 0.2 ) Rotate the image and ground truth to enforse the zero slop 
    y = (fixedgt(135,2)-fixedgt(115,2));
    x = (fixedgt(135,1)-fixedgt(115,1));
    current_slop = atand(double(y)/double(x));
      if current_slop ~= 0
          clear rotated_face
          origin=size(rgb2gray(fixedImage))/2+.5; % center of the whole image
          rotated_face = imrotate(fixedImage,current_slop,'bilinear','crop ') ;
          rotated_face(find(rotated_face(:,:) == 0)) = median(median(rgb2gray(fixedImage))); % blur the image
          rotated_gts  = rotate_points(fixedgt,current_slop,origin');
          fixedImage = rotated_face;
          fixedgt = rotated_gts;
          clear rotated_face rotated_gts
          disp('3.. Enforcing 0 slope over face')
      end
    
%%     1 a ) Detect faces : Chehra face detector here
        disp('4.. Chehra is working to detect noise :) ')
    [chbbox,chlandmarkPoints] = chehra68Detector(fixedImage,imagePath);
%%     1 b ) Detect faces : We will use zhu-ramanan face detector here 
%     disp('Zhu ramanan working')
%     [bbox,landmarkPoints] = zhuRamananDetector(fixedImage);
    % % I wont use chehra to frame the image so I commented in
%     try
%         
%         chface  =  fixedImage(chbbox(2):(chbbox(2)+chbbox(4)),chbbox(1):(chbbox(1)+chbbox(3)));
%     catch ME
%         if (strcmp(ME.identifier,'MATLAB:badsubscript'))
%             try 
%                 chface  =  fixedImage(chbbox(2):size(fixedImage,1),chbbox(1):(chbbox(1)+chbbox(3)));
%             catch
%                 chface  =  fixedImage(chbbox(2):size(fixedImage,1),chbbox(1):size(fixedImage,2));
%             end
%         end
%     end
% % I wont use zhu-ramanan face detector so I commented in
%     try
%           face    =  fixedImage(bbox(2):(bbox(2)+bbox(4)),bbox(1):(bbox(1)+bbox(3)));
%     catch ME
%         if (strcmp(ME.identifier,'MATLAB:badsubscript'))
%             try
%                 face    =  fixedImage(bbox(2):size(fixedImage,1),bbox(1):(bbox(1)+bbox(3)));
%             catch
%                 face    =  fixedImage(bbox(2):size(fixedImage,1),bbox(1):size(fixedImage,2));
%             end
%         end
%     end
%% 2 a) Crop only the face frame using ground truth data  
   disp('5..Cropping only the face out of the image')
    % specifiy borders :
    rightBound   = [fixedgt(41,1),fixedgt(41,2)];
    leftBound    = [fixedgt(1,1),fixedgt(1,2)];
    upperBound   = [(fixedgt(160,1)+fixedgt(180,1)),(fixedgt(160,2)+fixedgt(180,2))]/2;
    bottomBound  = [fixedgt(21,1),fixedgt(21,2)];
    
    % shift'em a little
    rightBound(1,1) = rightBound(1,1)+20;
    leftBound(1,1)  = leftBound(1,1)-20;
    upperBound(1,2) = upperBound(1,2)-20;
    bottomBound(1,2)= bottomBound(1,2)+10;
    % check borders
    if leftBound(1,1)<0  % left border
        leftBound(1,1) = 0;
    end
    if rightBound(1,1)>size(image,2) % right border
        rightBound(1,1) = size(image,2);
    end
    if upperBound(1,2)<0
        upperBound(1,2)= 0;
    end
    if   bottomBound(1,2) > size(image,1)
        bottomBound(1,2) = size(image,1);
    end
    gbbox= [leftBound(1,1),upperBound(1,2), rightBound(1,1)-leftBound(1,1),bottomBound(1,2)-upperBound(1,2)];
    gface  =  fixedImage(gbbox(2):(gbbox(2)+gbbox(4)),gbbox(1):(gbbox(1)+gbbox(3)));
%% 2 b) Shift ground truths accordingly  
    gshift = gbbox(1:2);
    fixedgt =  (horzcat((fixedgt(:,1)-gshift(1)),(fixedgt(:,2)-gshift(2))));
    chlandmarkPoints = (horzcat((chlandmarkPoints(:,1)-gshift(1)),(chlandmarkPoints(:,2)-gshift(2))));
%% 2 c ) Crop only the face frame using Zhu-Ramanan facial landmarks and shift gts accourdingly
%     shift = bbox(1:2);
%     fixedgt =  (horzcat((fixedgt(:,1)-shift(1)),(fixedgt(:,2)-shift(2))));
%     landmarkPoints =  (horzcat((landmarkPoints(:,1)-shift(1)),(landmarkPoints(:,2)-shift(2))));
% 2 d )  Crop only the face frame using Chehra facial landmarks and shift gts accourdingly
%     chshift = chbbox(1:2);
%     fixedgt =  (horzcat((fixedgt(:,1)-chshift(1)),(fixedgt(:,2)-chshift(2))));
%     chlandmarkPoints = (horzcat((chlandmarkPoints(:,1)-chshift(1)),(chlandmarkPoints(:,2)-chshift(2))));


    % 4 ) Extract 68 landmarks from 195 landmarks

    final_landmarks  = [fixedgt(1,:);fixedgt(4,:);fixedgt(5,:);fixedgt(7,:);fixedgt(10,:);...
        fixedgt(13,:);fixedgt(16,:);fixedgt(18,:);fixedgt(21,:);fixedgt(24,:);...
        fixedgt(26,:);fixedgt(27,:);fixedgt(29,:);fixedgt(32,:);fixedgt(35,:);fixedgt(38,:);...
        fixedgt(39,:);fixedgt(41,:);fixedgt(115,:);fixedgt(119,:);fixedgt(122,:);...
        fixedgt(126,:);fixedgt(129,:);fixedgt(132,:);fixedgt(135,:);fixedgt(139,:);...
        fixedgt(142,:);fixedgt(146,:);fixedgt(149,:);fixedgt(152,:);fixedgt(155,:);...
        (fixedgt(158,:)+fixedgt(172,:))/2;(fixedgt(161,:)+fixedgt(169,:))/2;...
        (fixedgt(163,:)+fixedgt(167,:))/2;fixedgt(165,:);fixedgt(175,:);...
        (fixedgt(178,:)+fixedgt(192,:))/2;(fixedgt(181,:)+fixedgt(189,:))/2;...
        (fixedgt(183,:)+fixedgt(187,:))/2;fixedgt(185,:);fixedgt(59,:);fixedgt(62,:);...
        fixedgt(64,:);fixedgt(66,:);fixedgt(68,:);fixedgt(70,:);fixedgt(73,:);...
        fixedgt(76,:);fixedgt(79,:);fixedgt(81,:);fixedgt(84,:);fixedgt(89,:);...
        fixedgt(92,:);fixedgt(95,:);fixedgt(98,:);fixedgt(100,:);fixedgt(105,:);...
        fixedgt(108,:);fixedgt(111,:);...
        chlandmarkPoints(28,:);chlandmarkPoints(29,:);chlandmarkPoints(30,:);chlandmarkPoints(31,:);chlandmarkPoints(32,:);...
        chlandmarkPoints(33,:);chlandmarkPoints(34,:);chlandmarkPoints(35,:);chlandmarkPoints(36,:)];


%     h = figure
%     subplot(2,2,1)
%     imshow(face);
%     hold on
%     plot(landmarkPoints(:,1),landmarkPoints(:,2),'r.','MarkerSize',10)
%     title('Subplot 1: Zhu-Ramanan Result')
% 
%     subplot(2,2,2)
%     imshow(face);
%     hold on
%     plot(fixedgt(:,1),fixedgt(:,2),'r.','MarkerSize',10)
%     title('Subplot 2: Ground-Truth Result')
% 
%     subplot(2,2,3)
%     imshow(chface);
%     hold on
%     plot(final_landmarks(:,1),final_landmarks(:,2),'r.','MarkerSize',10)
%     title('Subplot 3: Final Ground-Truth Result')
%     
%     subplot(2,2,4)
%     imshow(chface);
%     hold on
%     plot(chlandmarkPoints(:,1),chlandmarkPoints(:,2),'r.','MarkerSize',10)
%     title('Subplot 4: Chehra Ground-Truth Result')
    h = figure ;
    imshow(gface);
    hold on
    plot(final_landmarks(:,1),final_landmarks(:,2),'r.','MarkerSize',10)
    title(['image : ' num2str(i) ' name ' file_num ])
    
%     if  current_slop ~= 0
%         subplot(2,2,4)
%         imshow(rotated_face);
%         hold on
%         plot(rotated_gts(:,1),rotated_gts(:,2),'y.','MarkerSize',10)
%         title('Subplot 4: Final Ground-Truth Result with Rotation')
%     end
%     disp('press any key to continue');
%     pause;

saveas(h,char(strcat('results/',file_num,'_',subject_num, '.jpg')))

allGroundTruths(i,1) = {helenGroundTruth};
allGroundTruths(i,2) = {chlandmarkPoints};
allGroundTruths(i,3) = {final_landmarks};
% allGroundTruths(i,4) = {landmarkPoints};


    clear imagePath image face h landmarkPoints chlandmarkPoints final_landmarks
    close all;

  pause(1)
   catch ME
       fileID = fopen('process_log.txt','w');
fprintf(fileID,'%20s %50s\n %3d',char(strcat(file_num,'_',subject_num,'.jpg')),(ME.identifier),(ME.stack.line));
fclose(fileID);
       continue;
   end
end
disp('mkl');

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