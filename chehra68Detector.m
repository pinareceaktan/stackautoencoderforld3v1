function [bbox,landmarkPoints] = chehra68Detector(image,imagePath)

programRoot = pwd;
chehraPath  = 'C:\Users\FERA_ECE\Documents\MATLAB\chehra-68';
cd(chehraPath);

addpath(genpath('.'));

% 1 ) Face Detector : 0 for tree based model, 1 for viola jones
bbox_method = 0; 
% 2 ) Choose Fitting Views: 0 Two-Level and Multi-View Fitting, 1: Two-Level and Single-View Fitting
select_nview = 0; % Default : select_nview = 1 (suited for near-frontal faces)
% 3 ) Choose Fitting Method (1: DFRM, 2: GFRM-PO, 3: GFRM-Alternating
select_fitting = 1; %  Default : select_fitting = 1
% 4 ) Choose Visualize (0: Do Not Display Fitting Results, 1: Display Fitting Results and Pause of Inspection)
visualize=0;
    
        data(1).name    = imagePath ;
        data(1).img     = im2double(image);
        data(1).points  = []; % MAT containing 66 Landmark Locations

if select_fitting==1
    
    clm_model='model/DFRM_Model.mat';
    load(clm_model);    
    data=DFRM(clm_model,data,bbox_method,visualize);    
    
elseif select_fitting==2

    clm_model='model/GFRM_PO_Model.mat';
    load(clm_model);    
    data=GFRM_PO(CLM,grmf,data,bbox_method,select_nview,visualize);    
    
elseif select_fitting==3

    clm_model='model/GFRM_Alternating_Model.mat';
    disp('Loading Model...');pause(0.2);
    load(clm_model);    
    data=GFRM_Alt(CLM,grmf,data,bbox_method,select_nview,visualize);    
end
landmarkPoints = data(1).points;
 if size(landmarkPoints,1) >= 66
  % specifiy borders :
    rightBound   = [landmarkPoints(17,1),landmarkPoints(17,2)];
    leftBound    = [landmarkPoints(1,1),landmarkPoints(1,2)];
    upperBound   = [(landmarkPoints(20,1)+landmarkPoints(25,1)),(landmarkPoints(20,2)+landmarkPoints(25,2))]/2;
    bottomBound  = [landmarkPoints(9,1),landmarkPoints(9,2)];
    
   % shift'em a little
    
    rightBound(1,1) = rightBound(1,1)+20;
    leftBound(1,1)  = leftBound(1,1)-20;
    upperBound(1,2) = upperBound(1,2)-20;
    bottomBound(1,2)= bottomBound(1,2)+20;
    
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
    
    
 
    
    
    
    
    bbox= [leftBound(1,1),upperBound(1,2), rightBound(1,1)-leftBound(1,1),bottomBound(1,2)-upperBound(1,2)];
 else
        
          bbox = [0.1 0.1 size(image,1) size(image,2)];
 end

cd(programRoot)
end