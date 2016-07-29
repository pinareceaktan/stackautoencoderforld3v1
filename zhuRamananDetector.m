function [bbox,landmarkPoints] = zhuRamananDetector(image)
programRoot = pwd;
zhuRamananPath = 'C:\Users\FERA_ECE\Documents\MATLAB\zhu-ramanan';
% zhuRamananPath = 'C:\Users\DELL\Documents\MATLAB\zhu-ramanan';
cd(zhuRamananPath);
% compile; %dont compile every time
load('multipie_anil.mat'); % load the model
% 5 levels for each octave
model.interval = 5;
% set up the threshold
model.thresh = min(-0.65, model.thresh);

% define the mapping from view-specific mixture id to viewpoint
if length(model.components)==13 
    posemap = 90:-15:-90;
elseif length(model.components)==18
    posemap = [90:-15:15 0 0 0 0 0 0 -15:-15:-90];
else
    error('Can not recognize this model');
end

    bs = detect(image, model, model.thresh);
    bs = clipboxes(image, bs);
    bs = nms_face(bs,0.3);
    
    x1 = bs(1).xy(:,1);
    y1 = bs(1).xy(:,2);
    x2 = bs(1).xy(:,3);
    y2 = bs(1).xy(:,4);
    landmarkPoints = [(x1+x2)/2,(y1+y2)/2];
%     imshow(image);
%     hold on
% 	plot(landmarkPoints(:,1),landmarkPoints(:,2),'r.','MarkerSize',20);
    
    % specifiy borders :
    if size(landmarkPoints,1) >= 68
        rightBound   = [landmarkPoints(68,1),landmarkPoints(68,2)];
        leftBound    = [landmarkPoints(60,1),landmarkPoints(60,2)];
        upperBound   = [(landmarkPoints(19,1)+landmarkPoints(30,1)),landmarkPoints(19,2)+landmarkPoints(30,2)]/2;
        bottomBound  = [landmarkPoints(52,1),landmarkPoints(52,2)];
    
        % shift'em a little
        rightBound(1,1) = rightBound(1,1)+10;
        leftBound(1,1)  = leftBound(1,1)-5;
        upperBound(1,2) = upperBound(1,2)-5;
        bottomBound(1,2)= bottomBound(1,2)+10;

        bbox= [leftBound(1,1),upperBound(1,2), rightBound(1,1)-leftBound(1,1),bottomBound(1,2)-upperBound(1,2)];
    else 
          bbox = [0.1 0.1 size(image,1) size(image,2)];
    end
    
%     
%     % show highest scoring one
%     figure,showboxes(im, bs(1),posemap),title('Highest scoring detection');
%     % show all
%     figure,showboxes(im, bs,posemap),title('All detections above the threshold');
    cd(programRoot)