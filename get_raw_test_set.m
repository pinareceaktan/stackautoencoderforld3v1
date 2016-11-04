function [images,landmark_labels,pose_labels] = get_raw_data_set(how,pose_partition)
% This function gets three dataset and returns normalized or denormalized image arrays.
% Test Dataset 
%   300k - 600 images (300 indoor 300 outdoor)
imsize      = 50*50;
classsize   = 68*2;

%% Test Dataset :
programRoot = pwd;
testDataSetRoot     = strcat(programRoot,'\','datasetstuff','\TESTSET');
testFolder          = strcat(testDataSetRoot,'\testset');
testAnnotations     = strcat(testDataSetRoot,'\annotations');
dirContent          = dir(testAnnotations);
testImList          = cell(size(dirContent,1)-2,1);
testAnnotationList  = cell(size(dirContent,1)-2,1);

for i = 3 : size(dirContent,1)
    testImList(i-2,1)          =  strcat(regexpi(dirContent(i).name,'\S*(?=\.pts)','match'),'.png');
    testAnnotationList(i-2,1)  =  {dirContent(i).name};
end
% Create ground truth matrix for Test Matrix

% for i = 1:size(testAnnotationList,1)
%     try
%       fileID                = fopen(strcat(testAnnotations,'\',testAnnotationList{i,1}));
%       scannedText           = textscan(fileID,'%s');
%       testGroundTruth(i,1)  = testImList(i,1); 
% %     stupid extension because gts are like ass 
%       dim1 = scannedText{1,1}(6:2:140);
%       dim2 = scannedText{1,1}(7:2:141);
%     for j = 1 : size(dim1,1)
%         dim1arr(j,1)= str2num(dim1{j,1});
%         dim2arr(j,1)= str2num(dim2{j,1});
%     end
%     
%         testGroundTruth(i,2) = {[dim1arr dim2arr]};
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
% disp('Ground Truth is ready');
load('testGroundTruth.mat');

%% DATA Normalization

% for i = 1: size(testImList,1)
%     
%     try
%         imagePath       = strcat(testFolder,'\',testImList{i,1});
%         gt_index        = find(strcmp(testGroundTruth(:,1),testImList{i,1}));
%         image           = imread(imagePath);
%         disp(['1 Image: ' num2str(i) ' has loaded']);
%         groundTruth     = testGroundTruth{gt_index,2};
%         fixedImage      = image;
%         fixedgt         = groundTruth;
%         disp(['1.b Ground truth: ' num2str(i) ' has loaded']);
%         
%         image = fixedImage;
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
%           %% 4.1 ) Draw and save results
% %             h = figure ;
% %             imshow(gface);
% %             hold on
% %             plot(fixedgt(:,1),fixedgt(:,2),'r.','MarkerSize',10)
% %             title(['image : ' num2str(i)])
%         
% %             saveas(h,char(strcat('resultsTest/',num2str(i),'.png')))
% %             imwrite(gface,strcat(datasetAll,'/lfpw_',num2str(i)));
%             disp('4.1 Face and gts has saved visually')
%         %% 4.2 ) Final : Save denormalized data
%         testdatadenormalized(i).face            =  reshape(gface,imsize,1);
%         testdatadenormalized(i).groundtruth =  reshape(fixedgt,classsize,1);
%   
% 
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
%         fileID = fopen('logfile_test.txt','a');
%         fprintf(fileID,'%20s %40s %3d\n',num2str(i),(ME.identifier),(ME.stack.line));
%         fclose(fileID);
%         continue;
%     end
%     
% end

load('testdatadenormalized.mat');

for i = 1 : 600
    testdatadenormalized(i).pose = 1;
end
        pose_part = pose_partition;
%  pose_part=  0;
%  how = 'normalized';
switch how
    case 'normalized'
        [images,landmark_labels,pose_labels]=get_data_set_prepared(testdatadenormalized,pose_part);
        [gcnn_images,gcnnmean0std1_images] = gcn(images); 
        for i = 1:size(gcnn_images,2)
            gcnn_images(:,i) = mat2gray(gcnn_images(:,i));
            landmark_labels(:,i) = mat2gray(landmark_labels(:,i));
        end
        clear images
        images = gcnn_images;
        
       
    case 'denormalized'
         [images,landmark_labels,pose_labels]=get_data_set_prepared(testdatadenormalized,pose_part);

end

clear dirContent