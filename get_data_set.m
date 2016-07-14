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
addpath 'datasetstuff'
load('lfpwimages.mat', 'lfpwimages');

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
for i = 3:size(dirContent,1)
    counter = i-2;
    disp(counter);
    fileID = fopen(strcat(helenAnnotations,'\',dirContent(i).name));
    scannedText = textscan(fileID,'%d %c %d ');
    helenGroundTruth(counter,1) = {(strcat(num2str(scannedText{1,1}(1,1)),'_',num2str(scannedText{1,3}(1,1)),'.jpg'))}; 
    helenGroundTruth(counter,2) = {[scannedText{1,1}(2:end) scannedText{1,3}(2:end)]};
    clear fileID scannedText
end

clear dirContent


run_compilers;
for i = 1: size(trainImages,1)
    file_num        = regexpi(trainImages{i,1},'\d*(?=\_)','match');
    subject_num     = regexpi(trainImages{i,1},'(?<=_)\d*','match'); 
    imagePath       = strcat(helenTrainFolder,'\',trainImages{i,1},'.jpg');
    image       = imread(imagePath);
    disp(['image: ' num2str(i)]);
    % 1 ) Detect faces : We will use zhu-ramanan face detector here 
    [bbox,landmarkPoints] = zhuRamananDetector(image);
    disp('Zhu ramanan working')
    % 2 ) Extract only the faces and convert it to gray level
    face =  image(bbox(2):(bbox(2)+bbox(4)),bbox(1):(bbox(1)+bbox(3)));
    imshow(face);
    disp('Cropping face')
    % 3 ) Rotate face to enforce 0 slope 
    current_slop = (landmarkPoints(21,2)-landmarkPoints(10,2))/(landmarkPoints(21,1)-landmarkPoints(10,1));
    if current_slop ~= 0
    rotaed_face = imrotate(face,current_slop) ;
    disp('Enforcing 0 slope over face')
        imshow(rotaed_face);

    end
    

    clear imagePath image face 
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
% end
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


