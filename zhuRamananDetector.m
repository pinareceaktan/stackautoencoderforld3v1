function zhuRamananDetector(image)
programRoot = pwd;
zhuRamananPath = 'C:\Users\FERA_ECE\Documents\MATLAB\zhu-ramanan';
cd(zhuRamananPath);

compile;
load('multipie_an?l_p146.mat'); % load the model
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
    
    % show highest scoring one
    figure,showboxes(im, bs(1),posemap),title('Highest scoring detection');
    % show all
    figure,showboxes(im, bs,posemap),title('All detections above the threshold');
    
  % an?ldan landmark dönü?ümlerini al 




cd(programRoot);