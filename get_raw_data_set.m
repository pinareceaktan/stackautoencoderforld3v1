function [images,landmark_labels,pose_labels] = get_raw_data_set(how,pose_partition)
%% This function gets three dataset and returns normalized or denormalized
% image arrays.
%% Train Dataset
% 3 datasets is in use which explanied below
% the datasets are from i-bug. 
% Three databases: LFPW [21],Helen [22] and AFW [23]. 
% 811 images from LFPW,
% 2000 images from Helen
% 337 images from AFW as
% Note : To further augment the data, we conduct translation
% and rotation to each image.
%% Test Dataset 
%   300k - 600 images (300 indoor 300 outdoor)
%% Zhu-Ramanan Face Detector 
% X. Zhu, D. Ramanan. "Face detection, pose estimation and landmark localization in the wild" 
% Computer Vision and Pattern Recognition (CVPR) Providence, Rhode Island, June 2012. 
% website : https://www.ics.uci.edu/~xzhu/face/
imsize      = 50*50;
classsize   = 68*2;
programRoot = pwd; 
datasetAll  = strcat(programRoot,'\datasetstuff\ALL68');
%% LFPW Dataset :
lfpwDataSetRoot     = strcat(programRoot,'\','datasetstuff','\LFPW');
lfpwTrainFolder     = strcat(lfpwDataSetRoot,'\trainset');
lfpwAnnotations     = strcat(lfpwDataSetRoot,'\annotations');
dirContent          = dir(lfpwAnnotations);
lfpwtrainImList     = cell(size(dirContent,1)-2,1);
lfpwAnnotationList  = cell(size(dirContent,1)-2,1);

for i = 3 : size(dirContent,1)
    lfpwtrainImList(i-2,1)     =  strcat(regexpi(dirContent(i).name,'\S*(?=\.pts)','match'),'.png');
    lfpwAnnotationList(i-2,1)  =  {dirContent(i).name};
end
%% Create ground truth matrix for LFPW

% for i = 1:size(lfpwAnnotationList,1)
%     try
%       fileID                = fopen(strcat(lfpwAnnotations,'\',lfpwAnnotationList{i,1}));
%       scannedText           = textscan(fileID,'%s');
%       lfpwGroundTruth(i,1)  = lfpwtrainImList(i,1); 
% %     stupid extension because gts are like ass 
%       dim1 = scannedText{1,1}(6:2:140);
%       dim2 = scannedText{1,1}(7:2:141);
%     for j = 1 : size(dim1,1)
%         dim1arr(j,1)= str2num(dim1{j,1});
%         dim2arr(j,1)= str2num(dim2{j,1});
%     end
%     
%         lfpwGroundTruth(i,2) = {[dim1arr dim2arr]};
% 
%     fclose(fileID);
%     clear fileID scannedText dim1 dim2
%     catch ME
%            disp(num2str(i));
%         disp(ME.identifier);
%     continue;
%     end
% 
% end
% disp('sdfsdf');
load('lpfwGroundTruth.mat');

% %% DATA AUGMENTATION
% counter = 1;
% for i = 1: size(lfpwtrainImList,1)
%     
%     try
%         imagePath       = strcat(lfpwTrainFolder,'\',lfpwtrainImList{i,1});
%         gt_index        = find(strcmp(lfpwGroundTruth(:,1),lfpwtrainImList{i,1}));
%         image           = imread(imagePath);
%         disp(['1 Image: ' num2str(i) ' has loaded']);
%         groundTruth     = lfpwGroundTruth{gt_index,2};
%         fixedImage      = image;
%         fixedgt         = groundTruth;
%         disp(['1.b Ground truth: ' num2str(i) ' has loaded']);
%         
%         % 1 ) 
%         if size(fixedImage,3) ~= 1
%             fixedImage = rgb2gray(fixedImage);
%         end
%         lfpw_raw(counter).images = fixedImage;
%         lfpw_raw(counter).gt     = fixedgt; 
%         %% Data Augmentation : Rotate and Translate
%         % 2 ) Rotate image by -30
%         angle = -30;
%         [rotated_image,rotated_gt] = rotate_image(fixedImage,fixedgt,angle);
%         
%         counter = counter +1;
%         lfpw_raw(counter).images = rotated_image;
%         lfpw_raw(counter).gt     = rotated_gt; 
%         
%         clear rotated_image rotated_gt
%         
%         % 3) Rotate image by 30
%         clear angle rotated_image rotated_gt
%         angle = 30;
%         [rotated_image,rotated_gt] = rotate_image(fixedImage,fixedgt,angle);
%         
%         counter = counter +1;
%         lfpw_raw(counter).images = rotated_image;
%         lfpw_raw(counter).gt     = rotated_gt; 
%         
%         clear rotated_image rotated_gt
%         
%         % 4) Translate by +1 pxl in x
%         drift_arr =[1 0];
%         [translated_image,translated_gt] = translate_image(fixedImage,fixedgt,drift_arr);
%         
%         
%         counter = counter +1;
%         lfpw_raw(counter).images = translated_image;
%         lfpw_raw(counter).gt     = translated_gt; 
%        
%         clear translated_image translated_gt
%         
%         % 5) Translate by -1 pxl in x
%         drift_arr =[-1 0];
%         [translated_image,translated_gt] = translate_image(fixedImage,fixedgt,drift_arr);
%         
%         
%         counter = counter +1;
%         lfpw_raw(counter).images = translated_image;
%         lfpw_raw(counter).gt     = translated_gt; 
%         
%         clear translated_image translated_gt
%         
%         % 6) Translate by 2 pxl in x
%         clear drift_arr translated_image translated_gt ;
%         drift_arr =[2 0];
%         [translated_image,translated_gt] = translate_image(fixedImage,fixedgt,drift_arr);
%         
%         counter = counter +1;
%         lfpw_raw(counter).images = translated_image;
%         lfpw_raw(counter).gt     = translated_gt; 
%         
%         % 7) Translate by -2 pxl in x
%         clear drift_arr translated_image translated_gt ;
%         
%         drift_arr =[-2 0];
%         [translated_image,translated_gt] = translate_image(fixedImage,fixedgt,drift_arr);
%         
%         counter = counter +1;
%         lfpw_raw(counter).images = translated_image;
%         lfpw_raw(counter).gt     = translated_gt; 
%         
%         % 8) Translate by 1 pxl in y
%         clear drift_arr translated_image translated_gt ;
%         
%         drift_arr =[0 1];
%         [translated_image,translated_gt] = translate_image(fixedImage,fixedgt,drift_arr);
%         
%         counter = counter +1;
%         lfpw_raw(counter).images = translated_image;
%         lfpw_raw(counter).gt     = translated_gt; 
%                         
%         % 9) Translate by -1 pxl in y
%         clear drift_arr translated_image translated_gt ;
%         
%         drift_arr =[0 -1];
%         [translated_image,translated_gt] = translate_image(fixedImage,fixedgt,drift_arr);
%         
%         counter = counter +1;
%         lfpw_raw(counter).images = translated_image;
%         lfpw_raw(counter).gt     = translated_gt; 
%                 
%         % 10) Translate by 2 pxl in y
%         clear drift_arr translated_image translated_gt ;
%         
%         drift_arr =[0 2];
%         [translated_image,translated_gt] = translate_image(fixedImage,fixedgt,drift_arr);
%         
%         counter = counter +1;
%         lfpw_raw(counter).images = translated_image;
%         lfpw_raw(counter).gt     = translated_gt; 
%         
%         % 11) Translate by -2 pxl in y
%         
%         clear drift_arr translated_image translated_gt ;
%         
%         drift_arr =[0 -2];
%         [translated_image,translated_gt] = translate_image(fixedImage,fixedgt,drift_arr);
%         
%         counter = counter +1;
%         lfpw_raw(counter).images = translated_image;
%         lfpw_raw(counter).gt     = translated_gt; 
%                 
% %         % 12) Transform images : mirror
% %         tform                                    = affine2d([-1 0 0; 0 1 0; 0 0 1]);        
% %         counter = counter+1;
% %         lfpw_raw(counter).images     = imwarp(fixedImage,tform);
% %         lfpw_raw(counter).gt         = horzcat(size(fixedImage,2)-fixedgt(:,1),fixedgt(:,2));
%         counter = counter + 1;
%     catch ME
%         fileID = fopen('logfile_augmentation.txt','a');
%         fprintf(fileID,'%20s %40s %3d\n',['LFPW-' num2str(i)],(ME.identifier),(ME.stack.line));
%         fclose(fileID);
%         continue;
%     end
% end
% 
% %% Data Normalization
% for i = 1: size(lfpw_raw,2)
%     
%     fixedgt = lfpw_raw(i).gt;
%     fixedImage = lfpw_raw(i).images;
%     image = lfpw_raw(i).images;
%     try
%         %% Rotate to enforce zero slop
%         % Referance points
%         left_eye_xs = mean([fixedgt(38,1) fixedgt(39,1) fixedgt(41,1) fixedgt(42,1)]);
%         left_eye_ys = mean([fixedgt(38,2) fixedgt(39,2) fixedgt(41,2) fixedgt(42,2)]);
%         right_eye_xs = mean([fixedgt(44,1) fixedgt(45,1) fixedgt(47,1) fixedgt(48,1)]);
%         right_eye_ys = mean([fixedgt(44,2) fixedgt(45,2) fixedgt(47,2) fixedgt(48,2)]);
%         
%         y = (right_eye_ys-left_eye_ys);
%         x = (right_eye_xs-left_eye_xs);
%         
%         current_slop = atand(double(y)/double(x));
%         
%         if current_slop ~= 0
%             clear rotated_face % clear it if it was used before
%             if size(fixedImage,3) ~= 1
%                 fixedImage = rgb2gray(fixedImage); % gray
%             end
%             origin=size(fixedImage)/2+.5; % center of image
%             image_median = median(median(fixedImage))+40;
%             fixedImage(find(fixedImage(:,:) == 0)) = 9 ; % fixed image'in siyahlar?na saçma bir de?er ata
%             rotated_image = imrotate(fixedImage,current_slop,'bilinear','crop ') ;
%             rotated_image(find(rotated_image(:,:) == 0)) = image_median ; % kenardaki siyahlar? medyana boya
%             rotated_image(find(rotated_image(:,:) == 9)) = 0; % rotate etmeden siyah olan alanlar? siyaha boya
%             % Edge Tapering with gaussian filter
%             PSF = fspecial('gaussian',60,10);
%             edgesTapered = edgetaper(rotated_image,PSF);
%             
%             
%             rotated_gts  = rotate_points(fixedgt,current_slop,[origin(1,2) ; origin(1,1)]);
%             clear fixedImage fixedgt
%             fixedImage = edgesTapered;
%             fixedgt = rotated_gts;
%             clear rotated_face rotated_gts
%             disp('3.. Enforcing 0 slope over face')
%             
%         end
%         
%         
%         %% 2 a) Crop only the face frame using ground truth data
%         disp('2..Cropping only the face out of the image')
%         % specifiy borders :
%         lefters         = [1:6,18];
%         righters        = [13:17,27];
%         deepers         = 7:12;
%         leftUppers      = [1,2,18:22];
%         rightUppers     = [16,17,23:27];
%         
%         rightIndes    = find( fixedgt(:,1)==max(fixedgt(righters,1)));
%         leftIndes     = find( fixedgt(:,1)==min(fixedgt(lefters,1)));
%         leftTopIndes  = find( fixedgt(:,2)==min(fixedgt(leftUppers,2)));
%         rightTopIndes = find( fixedgt(:,2)==min(fixedgt(rightUppers,2)));
%         downIndes     = find( fixedgt(:,2)==max(fixedgt(deepers,2)));
%         
%         mostRightInd   = rightIndes(1,1);
%         mostLeftInd    = leftIndes(1,1);
%         mostLeftTopInd = leftTopIndes(1,1);
%         mostRightTopInd= rightTopIndes(1,1);
%         mostDownInd    = downIndes(1,1);
%         
%         rightBound   = [fixedgt(mostRightInd,1),fixedgt(mostRightInd,2)];
%         leftBound    = [fixedgt(mostLeftInd,1),fixedgt(mostLeftInd,2)];
%         upperBound   = [(fixedgt(mostLeftTopInd,1)+fixedgt(mostRightTopInd,1))/2,min(fixedgt(mostLeftTopInd,2),fixedgt(mostRightTopInd,2))];
%         bottomBound  = [fixedgt(mostDownInd,1),fixedgt(mostDownInd,2)];
%         
%         % calculate shifting amount
%         % Always shift 5% of the image, and precisely find that 5% is ?% of
%         % whole
%         face_width = rightBound(1,1)- leftBound(1,1);
%         im_width   = size(image,2);
%         x_shift = im_width*face_width*0.05/im_width;
%         face_height = bottomBound(2)-upperBound(2);
%         im_height   = size(image,1);
%         y_shift = im_height*face_height*0.05/im_height;
%         
%         % shift'em a little
%         rightBound(1,1) = rightBound(1,1)+x_shift;
%         leftBound(1,1)  = leftBound(1,1)-x_shift;
%         upperBound(1,2) = upperBound(1,2)-y_shift;
%         bottomBound(1,2)= bottomBound(1,2)+y_shift;
%         % check borders
%         if leftBound(1,1)<=0  % left border
%             leftBound(1,1) = 0.1;
%         end
%         if rightBound(1,1)>size(fixedImage,2) % right border
%             rightBound(1,1) = size(fixedImage,2);
%         end
%         if upperBound(1,2)<=0
%             upperBound(1,2)= 0.1;
%         end
%         if   bottomBound(1,2) > size(fixedImage,1)
%             bottomBound(1,2) = size(fixedImage,1);
%         end
%         gbbox= [leftBound(1,1),upperBound(1,2), rightBound(1,1)-leftBound(1,1),bottomBound(1,2)-upperBound(1,2)];
%         gface  =  fixedImage(gbbox(2):(gbbox(2)+gbbox(4)),gbbox(1):(gbbox(1)+gbbox(3)));
%         disp('2.a Face has extracted from image')
%         %% 2 b) Shift ground truths accordingly
%         gshift = double(gbbox(1:2));
%         fixedgt =  (horzcat((fixedgt(:,1)-gshift(1)),(fixedgt(:,2)-gshift(2))));
%         %% 3 ) Resize
%         nonresizedsize  = size(gface);
%         gface           = imresize(gface,[50 50]);
%         resizedsize     = size(gface);
%         scalex          = nonresizedsize(1)/resizedsize(1);
%         scaley          = nonresizedsize(1,2)/resizedsize(1,2);
%         fixedgt         = horzcat(fixedgt(:,1)/scaley,fixedgt(:,2)/scalex);
%         
%         disp('3 Resizing the face has done')
%         %  %% 4.1 ) Draw and save results
%             h = figure ;
%             imshow(gface);
%             hold on
%             plot(fixedgt(:,1),fixedgt(:,2),'r.','MarkerSize',10)
%             title(['image : ' num2str(i)])
%         
%             saveas(h,char(strcat('resultsLFPW/',num2str(i),'.png')))
% %             imwrite(gface,strcat(datasetAll,'/lfpw_',num2str(i)));
%             disp('4.1 Face and gts has saved visually')
%         %% 4.2 ) Final : Save denormalized data
%         lfpwdatadenormalized(i).face        =  reshape(gface,imsize,1);
%         lfpwdatadenormalized(i).groundtruth =  reshape(fixedgt,classsize,1);
%         %% 4.3 ) Normalization : Scale intensity values
%         gface   = im2double(gface);
%         gface   = normalizePic(gface);
%         fixedgt = normalizePic(fixedgt);
%         disp('4.2 scaling intensity values has done')
%         
%         %% 4.4 ) Final : Save data mat
%         lfpwdata(i).face        =  reshape(gface,imsize,1);
%         lfpwdata(i).groundtruth =  reshape(fixedgt,classsize,1);
%         %         prompt = 'What is the pose class? ';
%         %         x = input(prompt);
%         %         lfpwdata(i).pose = x;
%         disp(['4.3 ' num2str(i) ' saved to dataset']);
%         %% 5 ) Clear'em all
%         clear gt_index image groundTruth fixedImage fixedgt
%         clear lefters righters deepers leftUppers rightUppers
%         clear rightIndes leftIndes leftTopIndes rightTopIndes downIndes
%         clear mostRightInd mostLeftInd mostLeftTopInd mostRightTopInd mostDownInd
%         clear rightBound leftBound upperBound bottomBound
%         clear gbbox
%         clear gface gshift nonresizedsize resizedsize scalex scaley h
%         close all;
%         
%         pause(0.1)
%     catch ME
%         fileID = fopen('logfile_lfpw.txt','a');
%         fprintf(fileID,'%20s %40s %3d\n',num2str(i),(ME.identifier),(ME.stack.line));
%         fclose(fileID);
%         continue;
%     end
% end
% clear dirContent
% disp('LFPW done!')
%% open 
% load('lfpwdatadenormalized.mat')
load('lfpwdata.mat')
% %% 2ND AUGMENTATION : TRANSFORMATION 
% indx = size(lfpwdatadenormalized,2)+1;
% for i = 1: 11: size(lfpwdatadenormalized,2)
%      image	= reshape(lfpwdatadenormalized(i).face,50,50);
%      gt     = reshape(lfpwdatadenormalized(i).groundtruth,68,2);
% 
%      tform	= affine2d([-1 0 0; 0 1 0; 0 0 1]);   
%      warped_image = imwarp(image,tform);
%      lfpwdatadenormalized(indx).face                = reshape(warped_image,imsize,1) ;
%      warped_gt =  horzcat(50-gt(:,1),gt(:,2));
%      lfpwdatadenormalized(indx).groundtruth         = reshape(warped_gt,classsize,1);
%      indx = indx +1 ;
%     clear tform image gt warped_image warped_gt
% end
%         disp('2nd augmentation finalized')
% clear indx;

load('lfpwdatadenormalized.mat')

% %% AFW DATASET
% afwDataSetRoot = strcat(programRoot,'\','datasetstuff\AFW68'); 
% afwTrainFolder = strcat(afwDataSetRoot,'\trainset');
% afwAnnotations = strcat(afwDataSetRoot,'\annotations');
% dirContent     = dir(afwAnnotations);
% afwtrainImList     = cell(size(dirContent,1)-2,1);
% afwAnnotationList  = cell(size(dirContent,1)-2,1);
% for i = 3 : size(dirContent,1)
%     afwtrainImList(i-2,1)     =  strcat(regexpi(dirContent(i).name,'\S*(?=\.pts)','match'),'.jpg');
%     afwAnnotationList(i-2,1)  =  {dirContent(i).name};
% end
% % % Create ground truth matrix for AFW
% % for i = 1: size(afwAnnotationList,1)
% %     try
% %         fileID                  = fopen(strcat(afwAnnotations,'\',afwAnnotationList{i,1}));
% %         scannedText             = textscan(fileID,'%s');
% %         afwGroundTruth(i,1)     = afwtrainImList(i,1);
% %         %     stupid extension because gts are like ass
% %         dim1 = scannedText{1,1}(6:2:140);
% %         dim2 = scannedText{1,1}(7:2:141);
% %         for j = 1 : size(dim1,1)
% %             dim1arr(j,1)= str2num(dim1{j,1});
% %             dim2arr(j,1)= str2num(dim2{j,1});
% %         end
% %         afwGroundTruth(i,2) = {[dim1arr dim2arr]};
% %         fclose(fileID);
% %         clear fileID scannedText dim1 dim2
% %     catch ME
% %          disp(num2str(i));
% %          disp(ME.identifier);
% %          continue;
% %     end
% % end
% load('afwGroundTruth.mat')
% %% DATA AUGMENTATION
% 
% counter = 1;
% for i = 1: size(afwtrainImList,1)
%     
%     try
%         imagePath       = strcat(afwTrainFolder,'\',afwtrainImList{i,1});
%         gt_index        = find(strcmp(afwGroundTruth(:,1),afwtrainImList{i,1}));
%         image           = imread(imagePath);
%         disp(['1 Image: ' num2str(i) ' has loaded']);
%         groundTruth     = afwGroundTruth{gt_index,2};
%         fixedImage      = image;
%         fixedgt         = groundTruth;
%         disp(['1.b Ground truth: ' num2str(i) ' has loaded']);
%         % 1 )
%         if size(fixedImage,3) ~= 1
%             fixedImage = rgb2gray(fixedImage);
%         end
%         afw_raw(counter).images = fixedImage;
%         afw_raw(counter).gt     = fixedgt; 
%         %% Data Augmentation : Rotate and Translate
%         % 2 ) Rotate image by -30
%         angle = -30;
%         [rotated_image,rotated_gt] = rotate_image(fixedImage,fixedgt,angle);
%         
%         counter = counter +1;
%         afw_raw(counter).images = rotated_image;
%         afw_raw(counter).gt     = rotated_gt; 
%         
%         clear rotated_image rotated_gt
%         
%         % 3) Rotate image by 30
%         clear angle rotated_image rotated_gt
%         angle = 30;
%         [rotated_image,rotated_gt] = rotate_image(fixedImage,fixedgt,angle);
%         
%         counter = counter +1;
%         afw_raw(counter).images = rotated_image;
%         afw_raw(counter).gt     = rotated_gt; 
%         
%         clear rotated_image rotated_gt
%         
%         % 4) Translate by +1 pxl in x
%         drift_arr =[1 0];
%         [translated_image,translated_gt] = translate_image(fixedImage,fixedgt,drift_arr);
%         
%         
%         counter = counter +1;
%         afw_raw(counter).images = translated_image;
%         afw_raw(counter).gt     = translated_gt; 
%        
%         clear translated_image translated_gt
%         
%         % 5) Translate by -1 pxl in x
%         drift_arr =[-1 0];
%         [translated_image,translated_gt] = translate_image(fixedImage,fixedgt,drift_arr);
%         
%         
%         counter = counter +1;
%         afw_raw(counter).images = translated_image;
%         afw_raw(counter).gt     = translated_gt; 
%         
%         clear translated_image translated_gt
%         
%         % 6) Translate by 2 pxl in x
%         clear drift_arr translated_image translated_gt ;
%         drift_arr =[2 0];
%         [translated_image,translated_gt] = translate_image(fixedImage,fixedgt,drift_arr);
%         
%         counter = counter +1;
%         afw_raw(counter).images = translated_image;
%         afw_raw(counter).gt     = translated_gt; 
%         
%         % 7) Translate by -2 pxl in x
%         clear drift_arr translated_image translated_gt ;
%         
%         drift_arr =[-2 0];
%         [translated_image,translated_gt] = translate_image(fixedImage,fixedgt,drift_arr);
%         
%         counter = counter +1;
%         afw_raw(counter).images = translated_image;
%         afw_raw(counter).gt     = translated_gt; 
%         
%         % 8) Translate by 1 pxl in y
%         clear drift_arr translated_image translated_gt ;
%         
%         drift_arr =[0 1];
%         [translated_image,translated_gt] = translate_image(fixedImage,fixedgt,drift_arr);
%         
%         counter = counter +1;
%         afw_raw(counter).images = translated_image;
%         afw_raw(counter).gt     = translated_gt; 
%                         
%         % 9) Translate by -1 pxl in y
%         clear drift_arr translated_image translated_gt ;
%         
%         drift_arr =[0 -1];
%         [translated_image,translated_gt] = translate_image(fixedImage,fixedgt,drift_arr);
%         
%         counter = counter +1;
%         afw_raw(counter).images = translated_image;
%         afw_raw(counter).gt     = translated_gt; 
%                 
%         % 10) Translate by 2 pxl in y
%         clear drift_arr translated_image translated_gt ;
%         
%         drift_arr =[0 2];
%         [translated_image,translated_gt] = translate_image(fixedImage,fixedgt,drift_arr);
%         
%         counter = counter +1;
%         afw_raw(counter).images = translated_image;
%         afw_raw(counter).gt     = translated_gt; 
%         
%         % 11) Translate by -2 pxl in y
%         
%         clear drift_arr translated_image translated_gt ;
%         
%         drift_arr =[0 -2];
%         [translated_image,translated_gt] = translate_image(fixedImage,fixedgt,drift_arr);
%         
%         counter = counter +1;
%         afw_raw(counter).images = translated_image;
%         afw_raw(counter).gt     = translated_gt; 
%         
%         % 12) Transform images : mirror
% %         tform                                    = affine2d([-1 0 0; 0 1 0; 0 0 1]);
% %         transformed_image                        = imwarp(fixedImage,tform);
% %         
% %         counter = counter+1;
% %         afw_raw(counter).images     = transformed_image;
% %         afw_raw(counter).gt         = horzcat(size(fixedImage,2)-fixedgt(:,1),fixedgt(:,2));
%         counter = counter + 1;
%     catch ME
%         fileID = fopen('logfile_augmentation.txt','a');
%         fprintf(fileID,'%20s %40s %3d\n',[ 'AFW-' num2str(i)],(ME.identifier),(ME.stack.line));
%         fclose(fileID);
%         continue;
%     end
% end
% 
% %% DATA NORMALIZATION
% for i = 1:size(afw_raw,2)
%     disp([num2str(i) ' is running']);
%     try
%         fixedgt = afw_raw(i).gt;
%         fixedImage = afw_raw(i).images;
%         image = afw_raw(i).images;
%    
%         %% Rotate to enforce zero slop
%         % Referance points
%         left_eye_xs = mean([fixedgt(38,1) fixedgt(39,1) fixedgt(41,1) fixedgt(42,1)]);
%         left_eye_ys = mean([fixedgt(38,2) fixedgt(39,2) fixedgt(41,2) fixedgt(42,2)]);
%         right_eye_xs = mean([fixedgt(44,1) fixedgt(45,1) fixedgt(47,1) fixedgt(48,1)]);
%         right_eye_ys = mean([fixedgt(44,2) fixedgt(45,2) fixedgt(47,2) fixedgt(48,2)]);
%         
%         y = (right_eye_ys-left_eye_ys);
%         x = (right_eye_xs-left_eye_xs);
%         
%         current_slop = atand(double(y)/double(x));
%         
%         if current_slop ~= 0
%             clear rotated_face % clear it if it was used before
%             if size(fixedImage,3) ~= 1
%                 fixedImage = rgb2gray(fixedImage); % gray
%             end
%             origin=size(fixedImage)/2+.5; % center of image
%             image_median = median(median(fixedImage))+40;
%             fixedImage(find(fixedImage(:,:) == 0)) = 9 ; % fixed image'in siyahlar?na saçma bir de?er ata
%             rotated_image = imrotate(fixedImage,current_slop,'bilinear','crop ') ;
%             rotated_image(find(rotated_image(:,:) == 0)) = image_median ; % kenardaki siyahlar? medyana boya
%             rotated_image(find(rotated_image(:,:) == 9)) = 0; % rotate etmeden siyah olan alanlar? siyaha boya
%             % Edge Tapering with gaussian filter
%             PSF = fspecial('gaussian',60,10);
%             edgesTapered = edgetaper(rotated_image,PSF);
%             
%             
%             rotated_gts  = rotate_points(fixedgt,current_slop,[origin(1,2) ; origin(1,1)]);
%             clear fixedImage fixedgt
%             fixedImage = edgesTapered;
%             fixedgt = rotated_gts;
%             clear rotated_face rotated_gts
%             disp('3.. Enforcing 0 slope over face')
%             
%         end
%         
%         %% 2 a) Crop only the face frame using ground truth data
%         disp('2..Cropping only the face out of the image')
%         % specifiy borders :
%         lefters         = [1:6,18];
%         righters        = [13:17,27];
%         deepers         = 7:12;
%         leftUppers      = [1,2,18:22];
%         rightUppers     = [16,17,23:27];
%         
%         rightIndes    = find( fixedgt(:,1)==max(fixedgt(righters,1)));
%         leftIndes     = find( fixedgt(:,1)==min(fixedgt(lefters,1)));
%         leftTopIndes  = find( fixedgt(:,2)==min(fixedgt(leftUppers,2)));
%         rightTopIndes = find( fixedgt(:,2)==min(fixedgt(rightUppers,2)));
%         downIndes     = find( fixedgt(:,2)==max(fixedgt(deepers,2)));
%         
%         mostRightInd   = rightIndes(1,1);
%         mostLeftInd    = leftIndes(1,1);
%         mostLeftTopInd = leftTopIndes(1,1);
%         mostRightTopInd= rightTopIndes(1,1);
%         mostDownInd    = downIndes(1,1);
%         
%         rightBound   = [fixedgt(mostRightInd,1),fixedgt(mostRightInd,2)];
%         leftBound    = [fixedgt(mostLeftInd,1),fixedgt(mostLeftInd,2)];
%         upperBound   = [(fixedgt(mostLeftTopInd,1)+fixedgt(mostRightTopInd,1))/2,min(fixedgt(mostLeftTopInd,2),fixedgt(mostRightTopInd,2))];;
%         bottomBound  = [fixedgt(mostDownInd,1),fixedgt(mostDownInd,2)];
%         
%         % calculate shifting amount
%         % Always shift 5% of the image, and precisely find that 5% is ?% of
%         % whole
%         face_width = rightBound(1,1)- leftBound(1,1);
%         im_width   = size(image,2);
%         x_shift = im_width*face_width*0.05/im_width;
%         face_height = bottomBound(2)-upperBound(2);
%         im_height   = size(image,1);
%         y_shift = im_height*face_height*0.05/im_height;
%         
%         % shift'em a little
%         rightBound(1,1) = rightBound(1,1)+x_shift;
%         leftBound(1,1)  = leftBound(1,1)-x_shift;
%         upperBound(1,2) = upperBound(1,2)-y_shift;
%         bottomBound(1,2)= bottomBound(1,2)+y_shift;
%         % check borders
%         if leftBound(1,1)<=0  % left border
%             leftBound(1,1) = 0.1;
%         end
%         if rightBound(1,1)>size(fixedImage,2) % right border
%             rightBound(1,1) = size(fixedImage,2);
%         end
%         if upperBound(1,2)<=0
%             upperBound(1,2)= 0.1;
%         end
%         if   bottomBound(1,2) > size(fixedImage,1)
%             bottomBound(1,2) = size(fixedImage,1);
%         end
%         gbbox= [leftBound(1,1),upperBound(1,2), rightBound(1,1)-leftBound(1,1),bottomBound(1,2)-upperBound(1,2)];
%         gface  =  fixedImage(gbbox(2):(gbbox(2)+gbbox(4)),gbbox(1):(gbbox(1)+gbbox(3)));
%         disp('2.a Face has extracted from image')
%         %% 2 b) Shift ground truths accordingly
%         gshift = double(gbbox(1:2));
%         fixedgt =  (horzcat((fixedgt(:,1)-gshift(1)),(fixedgt(:,2)-gshift(2))));
%         %% 3 ) Resize
%         nonresizedsize  = size(gface);
%         gface           = imresize(gface,[50 50]);
%         resizedsize     = size(gface);
%         scalex          = nonresizedsize(1)/resizedsize(1);
%         scaley          = nonresizedsize(1,2)/resizedsize(1,2);
%         fixedgt         = horzcat(fixedgt(:,1)/scaley,fixedgt(:,2)/scalex);
%         
%         disp('3 Resizing the face has done')
%         %  %% 4.1 ) Draw and save results
% %         h = figure ;
% %         imshow(gface);
% %         hold on
% %         plot(fixedgt(:,1),fixedgt(:,2),'r.','MarkerSize',10)
% %         title(['image : ' num2str(i)])
% %         saveas(h,char(strcat('resultsAFW/',num2str(i))))
%         %     imwrite(gface,strcat(datasetAll,'/afw_',num2str(i)));
%         disp('4.1 Face and gts has saved visually')
%         %% 4.2 ) Final : Save denormalized data
%         afwdatadenormalized(i).face        =  reshape(gface,imsize,1);
%         afwdatadenormalized(i).groundtruth =  reshape(fixedgt,classsize,1);
%         %% 4.3 ) Normalization : Scale intensity values
%         gface   =  im2double(gface);
%         gface   = normalizePic(gface);
%         fixedgt = normalizePic(fixedgt);
%         disp('4.2 scaling intensity values has done')
%         %% 4.4 ) Final : Save data mat
%         afwdata(i).face        =  reshape(gface,imsize,1);
%         afwdata(i).groundtruth =  reshape(fixedgt,classsize,1);
%         %         prompt = 'What is the pose class? ';
%         %         xy = input(prompt);
%         %         afwdata(i).pose = xy;
%         %         disp(['4.3 ' num2str(i) ' saved to dataset']);
%         %% 5 ) Clear'em all
%         
%         clear gt_index image groundTruth fixedImage fixedgt
%         clear lefters righters deepers leftUppers rightUppers
%         clear rightIndes leftIndes leftTopIndes rightTopIndes downIndes
%         clear mostRightInd mostLeftInd mostLeftTopInd mostRightTopInd mostDownInd
%         clear rightBound leftBound upperBound bottomBound
%         clear gbbox
%         clear gface gshift nonresizedsize resizedsize scalex scaley h
%         close all;
%         close all;
%         
%         pause(0.1);
%     catch ME
%         fileID = fopen('logfile_afw.txt','a');
%         fprintf(fileID,'%20s %40s %3d\n',num2str(i),(ME.identifier),(ME.stack.line));
%         fclose(fileID);
%         continue;
%     end
% end
% clear dirContent
% disp('AFW is Ready');
% % % Open
load('afwdata.mat')
% load('afwdatadenormalized.mat')

%% 2ND AUGMENTATION : TRANSFORMATION 
% indx = size(afwdatadenormalized,2)+1;
% for i = 1: 11: size(afwdatadenormalized,2)
%      image	= reshape(afwdatadenormalized(i).face,50,50);
%      gt     = reshape(afwdatadenormalized(i).groundtruth,68,2);
% 
%      tform	= affine2d([-1 0 0; 0 1 0; 0 0 1]);   
%      warped_image = imwarp(image,tform);
%      afwdatadenormalized(indx).face                = reshape(warped_image,imsize,1) ;
%      warped_gt =  horzcat(50-gt(:,1),gt(:,2));
%      afwdatadenormalized(indx).groundtruth         = reshape(warped_gt,classsize,1);
%      indx = indx +1 ;
%     clear tform image gt warped_image warped_gt
% end
%         disp('2nd augmentation finalized')
% clear indx;
load('afwdatadenormalized.mat')

%% HELEN DATASET
% helenDataSetRoot = strcat(programRoot,'\','datasetstuff\HELEN68'); 
% helenTrainFolder = strcat(helenDataSetRoot,'\trainset');
% helenAnnotations = strcat(helenDataSetRoot,'\annotations');
% helentrainImList = strcat(helenDataSetRoot,'\train_im_list.txt');
% fileID           = fopen(helentrainImList);
% scannedText = textscan(fileID,'%s');
% trainImages = scannedText{1,1};
% %% Create ground truth matrix for Helen
% 
% % dirContent  =   dir(helenAnnotations);
% % for i = 3:size(dirContent,1)
% %     try
% %     counter = i-2;
% %     fileID = fopen(strcat(helenAnnotations,'\',dirContent(i).name));
% %     scannedText = textscan(fileID,'%s');
% %     helenGroundTruth(counter,1) = (strcat(regexpi(dirContent(i).name,'\S*(?=\.pts)','match'),'.jpg')); 
% %     stupid extension because gts are like ass 
% %     dim1 = scannedText{1,1}(6:2:140);
% %     dim2 = scannedText{1,1}(7:2:141);
% %     for j = 1 : size(dim1,1)
% %         dim1arr(j,1)= str2num(dim1{j,1});
% %         dim2arr(j,1)= str2num(dim2{j,1});
% %     end
% % %     helenGroundTruth(counter,2) = {[scannedText{1,1}(6:2:140) scannedText{1,1}(7:2:141)]};
% %     helenGroundTruth(counter,2) = {[dim1arr dim2arr]};
% % 
% %     fclose(fileID);
% %     clear fileID scannedText counter dim1 dim2 
% %     catch ME
% %         disp(num2str(i));
% %         disp(ME.identifier);
% %     continue;
% %     end
% % end
% % fclose all
% % clear dirContent
% load('helenGroundTruthRaw.mat') % load ground-truth matrix
% %% DATA AUGMENTATION
% clear counter
% counter = 1;
% for i = 1: size(trainImages,1)
%     
%     try
%         file_num        = regexpi(trainImages{i,1},'\d*(?=\_)','match');
%         subject_num     = regexpi(trainImages{i,1},'(?<=_)\d*','match');
%         imagePath       = strcat(helenTrainFolder,'\',trainImages{i,1},'.jpg');
%         gt_index        = find(strcmp(helenGroundTruth(:,1),strcat(file_num,'_',subject_num,'.jpg')));
%         image           = imread(imagePath);
%         disp(['1 Image: ' num2str(i) ' has loaded']);
%         groundTruth     = helenGroundTruth{gt_index,2};
%         fixedImage      = image;
%         fixedgt         = groundTruth;
%         disp(['1.b Ground truth: ' num2str(i) ' has loaded']);
%         % 1 )
%          if size(fixedImage,3) ~= 1
%             fixedImage = rgb2gray(fixedImage);
%         end
%         helen_raw(counter).images = fixedImage;
%         helen_raw(counter).gt     = fixedgt; 
%         %% Data Augmentation : Rotate and Translate
%         % 2 ) Rotate image by -30
%         angle = -30;
%         [rotated_image,rotated_gt] = rotate_image(fixedImage,fixedgt,angle);
%         
%         counter = counter +1;
%         helen_raw(counter).images = rotated_image;
%         helen_raw(counter).gt     = rotated_gt; 
%         
%         clear rotated_image rotated_gt
%         
%         % 3) Rotate image by 30
%         clear angle rotated_image rotated_gt
%         angle = 30;
%         [rotated_image,rotated_gt] = rotate_image(fixedImage,fixedgt,angle);
%         
%         counter = counter +1;
%         helen_raw(counter).images = rotated_image;
%         helen_raw(counter).gt     = rotated_gt; 
%         
%         clear rotated_image rotated_gt
%         
%         % 4) Translate by +1 pxl in x
%         drift_arr =[1 0];
%         [translated_image,translated_gt] = translate_image(fixedImage,fixedgt,drift_arr);
%         
%         
%         counter = counter +1;
%         helen_raw(counter).images = translated_image;
%         helen_raw(counter).gt     = translated_gt; 
%        
%         clear translated_image translated_gt
%         
%         % 5) Translate by -1 pxl in x
%         drift_arr =[-1 0];
%         [translated_image,translated_gt] = translate_image(fixedImage,fixedgt,drift_arr);
%         
%         
%         counter = counter +1;
%         helen_raw(counter).images = translated_image;
%         helen_raw(counter).gt     = translated_gt; 
%         
%         clear translated_image translated_gt
%         
%         % 6) Translate by 2 pxl in x
%         clear drift_arr translated_image translated_gt ;
%         drift_arr =[2 0];
%         [translated_image,translated_gt] = translate_image(fixedImage,fixedgt,drift_arr);
%         
%         counter = counter +1;
%         helen_raw(counter).images = translated_image;
%         helen_raw(counter).gt     = translated_gt; 
%         
%         % 7) Translate by -2 pxl in x
%         clear drift_arr translated_image translated_gt ;
%         
%         drift_arr =[-2 0];
%         [translated_image,translated_gt] = translate_image(fixedImage,fixedgt,drift_arr);
%         
%         counter = counter +1;
%         helen_raw(counter).images = translated_image;
%         helen_raw(counter).gt     = translated_gt; 
%         
%         % 8) Translate by 1 pxl in y
%         clear drift_arr translated_image translated_gt ;
%         
%         drift_arr =[0 1];
%         [translated_image,translated_gt] = translate_image(fixedImage,fixedgt,drift_arr);
%         
%         counter = counter +1;
%         helen_raw(counter).images = translated_image;
%         helen_raw(counter).gt     = translated_gt; 
%                         
%         % 9) Translate by -1 pxl in y
%         clear drift_arr translated_image translated_gt ;
%         
%         drift_arr =[0 -1];
%         [translated_image,translated_gt] = translate_image(fixedImage,fixedgt,drift_arr);
%         
%         counter = counter +1;
%         helen_raw(counter).images = translated_image;
%         helen_raw(counter).gt     = translated_gt; 
%                 
%         % 10) Translate by 2 pxl in y
%         clear drift_arr translated_image translated_gt ;
%         
%         drift_arr =[0 2];
%         [translated_image,translated_gt] = translate_image(fixedImage,fixedgt,drift_arr);
%         
%         counter = counter +1;
%         helen_raw(counter).images = translated_image;
%         helen_raw(counter).gt     = translated_gt; 
%         
%         % 11) Translate by -2 pxl in y
%         
%         clear drift_arr translated_image translated_gt ;
%         
%         drift_arr =[0 -2];
%         [translated_image,translated_gt] = translate_image(fixedImage,fixedgt,drift_arr);
%         
%         counter = counter +1;
%         helen_raw(counter).images = translated_image;
%         helen_raw(counter).gt     = translated_gt; 
%         
%         % 12) Transform images : mirror
% %         tform                                    = affine2d([-1 0 0; 0 1 0; 0 0 1]);
% %         transformed_image                        = imwarp(fixedImage,tform);
% %         
% %         counter = counter+1;
% %         helen_raw(counter).images     = transformed_image;
% %         helen_raw(counter).gt         = horzcat(size(fixedImage,2)-fixedgt(:,1),fixedgt(:,2));
%         counter = counter+1;
%     catch ME
%         fileID = fopen('logfile_augmentation.txt','a');
%         fprintf(fileID,'%20s %40s %3d\n',[ 'HELEN-' num2str(i)],(ME.identifier),(ME.stack.line));
%         fclose(fileID);
%         continue;
%     end
% end
% 
% %% DATA NORMALIZATION
% for i =1: size(helen_raw,2)
%     disp([num2str(i) ' is running']);
%     
%     try
%         fixedgt = helen_raw(i).gt;
%         fixedImage = helen_raw(i).images;
%         image = helen_raw(i).images;
% %% Rotate to enforce zero slop
% % Referance points
% left_eye_xs = mean([fixedgt(38,1) fixedgt(39,1) fixedgt(41,1) fixedgt(42,1)]);
% left_eye_ys = mean([fixedgt(38,2) fixedgt(39,2) fixedgt(41,2) fixedgt(42,2)]);
% right_eye_xs = mean([fixedgt(44,1) fixedgt(45,1) fixedgt(47,1) fixedgt(48,1)]);
% right_eye_ys = mean([fixedgt(44,2) fixedgt(45,2) fixedgt(47,2) fixedgt(48,2)]);
% 
% y = (right_eye_ys-left_eye_ys);
% x = (right_eye_xs-left_eye_xs);
% 
% current_slop = atand(double(y)/double(x));
%  
%      if current_slop ~= 0
%           clear rotated_face % clear it if it was used before
%           if size(fixedImage,3) ~= 1
%           fixedImage = rgb2gray(fixedImage); % gray
%           end
%           origin=size(fixedImage)/2+.5; % center of image
%           image_median = median(median(fixedImage))+40;
%           fixedImage(find(fixedImage(:,:) == 0)) = 9 ; % fixed image'in siyahlar?na saçma bir de?er ata
%           rotated_image = imrotate(fixedImage,current_slop,'bilinear','crop ') ;
%           rotated_image(find(rotated_image(:,:) == 0)) = image_median ; % kenardaki siyahlar? medyana boya
%           rotated_image(find(rotated_image(:,:) == 9)) = 0; % rotate etmeden siyah olan alanlar? siyaha boya 
%           % Edge Tapering with gaussian filter
%           PSF = fspecial('gaussian',60,10);
%           edgesTapered = edgetaper(rotated_image,PSF);
% 
% 
%           rotated_gts  = rotate_points(fixedgt,current_slop,[origin(1,2) ; origin(1,1)]);
%           clear fixedImage fixedgt
%           fixedImage = edgesTapered;
%           fixedgt = rotated_gts;
%           clear rotated_face rotated_gts
%           disp('3.. Enforcing 0 slope over face')
%  
%       end
% %% 2 a) Crop only the face frame using ground truth data  
%    disp('2..Cropping only the face out of the image')
%     % specifiy borders :
%     lefters         = [1:6,18];
%     righters        = [13:17,27];
%     deepers         = 7:12;
%     leftUppers      = [1,2,18:22];
%     rightUppers     = [16,17,23:27];
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
%     upperBound   = [(fixedgt(mostLeftTopInd,1)+fixedgt(mostRightTopInd,1))/2,min(fixedgt(mostLeftTopInd,2),fixedgt(mostRightTopInd,2))];
%     bottomBound  = [fixedgt(mostDownInd,1),fixedgt(mostDownInd,2)];
%   
%     % calculate shifting amount
%     % Always shift 5% of the image, and precisely find that 5% is ?% of
%     % whole
%     face_width = rightBound(1,1)- leftBound(1,1);
%     im_width   = size(image,2);
%     x_shift = im_width*face_width*0.05/im_width;
%     face_height = bottomBound(2)-upperBound(2);
%     im_height   = size(image,1);
%     y_shift = im_height*face_height*0.05/im_height;
% 
%     % shift'em a little
%     rightBound(1,1) = rightBound(1,1)+x_shift;
%     leftBound(1,1)  = leftBound(1,1)-x_shift;
%     upperBound(1,2) = upperBound(1,2)-y_shift;
%     bottomBound(1,2)= bottomBound(1,2)+y_shift;
%     % check borders
%     if leftBound(1,1)<=0  % left border
%         leftBound(1,1) = 0.1;
%     end
%     if rightBound(1,1)>size(fixedImage,2) % right border
%         rightBound(1,1) = size(fixedImage,2);
%     end
%     if upperBound(1,2)<=0
%         upperBound(1,2)= 0.1;
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
%     gface           = imresize(gface,[50 50]);
%     resizedsize     = size(gface);
%     scalex          = nonresizedsize(1)/resizedsize(1);
%     scaley          = nonresizedsize(1,2)/resizedsize(1,2);
%     fixedgt         = horzcat(fixedgt(:,1)/scaley,fixedgt(:,2)/scalex);
%     
%     disp('3 Resizing the face has done')
% %% 4.1 ) Draw and save results
% %     h = figure ;
% %     imshow(gface);
% %     hold on
% %     plot(fixedgt(:,1),fixedgt(:,2),'r.','MarkerSize',10)
% %     title(['image : ' num2str(i)])   
% %     saveas(h,char(strcat('resultsHELEN/',file_num,'_',subject_num, '.jpg')))
% %     
% %     imwrite(gface,char(strcat(datasetAll,'/helen_',file_num,'_',subject_num, '.jpg')));
% %     disp('4.1 Face and gts has saved visually')
% %% 4.2 ) Final : Save denormalized data
% 	helendatadenormalized(i).face        =  reshape(gface,imsize,1);
%     helendatadenormalized(i).groundtruth =  reshape(fixedgt,classsize,1);
% %% 4.3 ) Normalization : Scale intensity values
%     gface   = im2double(gface);
%     gface   = normalizePic(gface);
%     fixedgt = normalizePic(fixedgt);
%     disp('4.2 scaling intensity values has done')
% %% 4.4 ) Final : Save data mat
%         helendata(i).face        =  reshape(gface,imsize,1);
%         helendata(i).groundtruth =  reshape(fixedgt,classsize,1);
% %         prompt = 'What is the pose class? ';
% %         xyz = input(prompt);
% %         helendata(i).pose = xyz;
%         disp(['4.3 ' num2str(i) ' saved to dataset']);
%        
% 
%  %% 5 ) Clear'em all       
%         clear gt_index image groundTruth fixedImage fixedgt
%         clear lefters righters deepers leftUppers rightUppers
%         clear rightIndes leftIndes leftTopIndes rightTopIndes downIndes
%         clear mostRightInd mostLeftInd mostLeftTopInd mostRightTopInd mostDownInd
%         clear rightBound leftBound upperBound bottomBound
%         clear gbbox
%         clear gface gshift nonresizedsize resizedsize scalex scaley h
%         close all;
%     
%   pause(0.1)
%    catch ME
%        fileID = fopen('logfile_helen.txt','a');
%        fprintf(fileID,'%20s %40s %3d\n',char(strcat(file_num,'_',subject_num,'.jpg')),(ME.identifier),(ME.stack.line));
%        fclose(fileID);
%        continue;
%    end
% end
% disp('Helen is Ready !');
%% open
load('helendata.mat')
% load('helendatadenormalized.mat')
% 
% %% 2ND AUGMENTATION : TRANSFORMATION 
% indx = size(helendatadenormalized,2)+1;
% for i = 1: 11: size(helendatadenormalized,2)
%      image	= reshape(helendatadenormalized(i).face,50,50);
%      gt     = reshape(helendatadenormalized(i).groundtruth,68,2);
% 
%      tform	= affine2d([-1 0 0; 0 1 0; 0 0 1]);   
%      warped_image = imwarp(image,tform);
%      helendatadenormalized(indx).face                = reshape(warped_image,imsize,1) ;
%      warped_gt =  horzcat(50-gt(:,1),gt(:,2));
%      helendatadenormalized(indx).groundtruth         = reshape(warped_gt,classsize,1);
%      indx = indx +1 ;
%     clear tform image gt warped_image warped_gt
% end
%         disp('2nd augmentation finalized')
% clear indx;
load('helendatadenormalized.mat')

% load('datadenormalized.mat');
%% Data Matrix : concatanate them 
% data                = [lfpwdata,afwdata,helendata];
% datadenormalized    = [lfpwdatadenormalized afwdatadenormalized helendatadenormalized];

% for i = 1: size(datadenormalized,2)
% % test_index = randperm(size(datadenormalized,2),350);
% % for i = 1:size(test_index,2)
%     disp(i);
%     image           = reshape(datadenormalized(i).face,50,50);
%     landmarks68     = reshape(datadenormalized(i).groundtruth,68,2);
%     landmarks66     = landmarks68(1:66,:);
%     [yaw,pitch,roll]= chehra_pose_estimator(image,landmarks66);
% %   disp(['sol-sag : ' num2str(yaw) ', asagi-yukari : ' num2str(pitch)] );
% 

%% Pose Classification Step 1 : Chehra
% % 1 Horizontal Class
% yaw_sign =  sign(yaw);
% yaw_val  =  abs(yaw);
% 
% if yaw_val > 8.4
%     switch yaw_sign 
%         case -1
%             horizontal_pose = 1;
%         case 1
%              horizontal_pose = -1;
%     end
% else 
%     horizontal_pose = 0;
% end
% 
% % 2 Vertical Class  
% pitch_sign = sign(pitch);
% pitch_val  = abs(pitch);
% if pitch_val >19
%    switch pitch_sign 
%         case -1
%             vertical_pose = 1;
%        case 1
%              vertical_pose = -1;
%    end
% else
%     vertical_pose = 0;
% end
%   
% datadenormalized(i).pose = [horizontal_pose vertical_pose];
% datadenormalized(i).yawpitch = [yaw pitch]; 
% %%  Plot test images: decide parameters
% %     h = figure ;
% %     imshow(image);
% %     hold on
% %     plot(landmarks68(:,1),landmarks68(:,2),'b*');
% %     title([num2str(i),' : Pitch = ',num2str(pitch),' ; Yaw = ',num2str(yaw)]);
% %     set(gcf,'units','normalized','outerposition',[0 0 1 1]);
% %     saveas(h,char(strcat('resultsPoseEstimation','/posetest_',num2str(i), '.jpg')));
% %     pause(0.1);
% %     close all;
% %     clear image landmarks68 landmarks66
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% end
% disp('done');

%% Pose Classification Step 2 : Manuel

% for i = 1: 12:size(datadenormalized,2)
%         % test_index = randperm(size(datadenormalized,2),350);
%         % for i = [2293,6035,7812,8988,11569,21864,23041,24288,25416,33864,34861,36541]
%         % 	checkpose = datadenormalized(i).pose ;
%         %     if  ~isnumeric(checkpose(1)) ||  ~isnumeric(checkpose(2))
% 
%     disp(i);
%     image   = reshape(datadenormalized(i).face,50,50);
%    
%     imshow(image);
%     yawpitch = datadenormalized(i).yawpitch;
% %     disp(['yaw: ' num2str(yawpitch(1)) ',pitch: ' num2str(yawpitch(2))] );
% %     disp(['yaw class: ' num2str(datadenormalized(i).pose(1))]);
%     prompt=('What is yaw class?');
%     yaw_class = input(prompt,'s');
%     switch yaw_class
%         case 'k'
%             yaw_class = 1;
%         case 'þ'
%             yaw_class = -1;
%         case 'l'
%             yaw_class = 0;
%     end
%     
% %     disp(['pitch class: ' num2str(datadenormalized(i).pose(2))]);
%     prompt = ('What is pitch class?');
%     pitch_class = input(prompt,'s');
%     switch pitch_class
%         case 'o'
%             pitch_class = -1;
%         case 'ç'
%             pitch_class = 1;
%         case 'l'
%             pitch_class = 0;
%     end
%     
%     datadenormalized(i+1).pose = [yaw_class pitch_class];
%     datadenormalized(i+2).pose = [yaw_class pitch_class];
%     datadenormalized(i+3).pose = [yaw_class pitch_class];
%     datadenormalized(i+4).pose = [yaw_class pitch_class];
%     datadenormalized(i+5).pose = [yaw_class pitch_class];
%     datadenormalized(i+6).pose = [yaw_class pitch_class];
%     datadenormalized(i+7).pose = [yaw_class pitch_class];
%     datadenormalized(i+8).pose = [yaw_class pitch_class];
%     datadenormalized(i+9).pose = [yaw_class pitch_class];
%     datadenormalized(i+10).pose = [yaw_class pitch_class];
%     datadenormalized(i+11).pose = [-1*yaw_class pitch_class];
% %     end
% end



load('datadenormalized.mat')

augmented_data_denormalized = datadenormalized;


disp('Augmentation is done!')

switch how
    case 'NotNormalized' 
        %% mean 0 std 1 normalization
        pose_part = pose_partition;
        [images,landmark_labels,pose_labels]=get_data_set_prepared(augmented_data_denormalized,pose_part);
    case 'GCNNormalized'
        pose_part = pose_partition;
        [images,landmark_labels,pose_labels]=get_data_set_prepared(augmented_data_denormalized,pose_part);
        %% STP1: Global Contrast Normalization
        [gcnn_images,gcnnmean0std1_images] = gcn(images); 
        clear images
        images = gcnn_images;
        
    case 'GCNMean0Std1Normalized'
        pose_part = pose_partition;
        [images,landmark_labels,pose_labels]=get_data_set_prepared(augmented_data_denormalized,pose_part);
        %% STP1: Global Contrast Normalization
        [gcnn_images,gcnnmean0std1_images] = gcn(images); 
        clear images
        images = gcnnmean0std1_images;
        landmark_labels = normalizePic(landmark_labels);
    case 'GCNZeroOneNormalization'
        pose_part = pose_partition;
        [images,landmark_labels,pose_labels]=get_data_set_prepared(augmented_data_denormalized,pose_part);
        %% STP1: Global Contrast Normalization
        [gcnn_images,gcnnmean0std1_images] = gcn(images); 
        for i = 1:size(gcnn_images,2)
            gcnn_images(:,i) = mat2gray(gcnn_images(:,i));
            landmark_labels(:,i) = mat2gray(landmark_labels(:,i));
        end
        clear images
        images = gcnn_images;
        
        
    case 'classicalNormalization'
         pose_part = pose_partition;
        [images,landmark_labels,pose_labels]=get_data_set_prepared(augmented_data_denormalized,pose_part);
        for i = 1:size(images,2)
            images(:,i) = mat2gray(images(:,i));
            landmark_labels(:,i) = mat2gray(landmark_labels(:,i));
        end
        
end
disp('Train set is READY !');
cd(programRoot);