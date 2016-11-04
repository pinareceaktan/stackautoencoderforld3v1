% error
load('landmark_prd.mat') ;
load('testdatadenormalized.mat');

% for i = 1: size(testdatadenormalized,2)
%     gt(:,:,i) = [reshape(testdatadenormalized(i).groundtruth,68,2)];
%     predictions(:,:,i) = [reshape(landmark_prd(:,i),68,2)];
% end

%  [ error_per_image ] = compute_error( gt, predictions);
 
 load('error_per_image.mat');
 err_mat(:,1) = numel(find(error_per_image(:)>0.1));
 err_mat(:,2) = numel(find(error_per_image(:)>0.15));
 err_mat(:,3) = numel(find(error_per_image(:)>0.2));
