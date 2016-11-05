%% show results on test set
% load('landmark_prd.mat') ;
% load('pose_prd.mat');
% load('testdatadenormalized.mat');
% root = pwd;
% for i = 1: size(testdatadenormalized,2)
%     validation = testdatadenormalized(i).groundtruth;
%     fig = figure;
%     subplot(1,2,1);
%     imshow(reshape(testdatadenormalized(i).face,50,50));
%     hold on;
%     plot(landmark_prd(1:68,i),landmark_prd(69:136,i),'r.','MarkerSize',20);
%     title('My Predictions')
%     subplot(1,2,2);
%     imshow(reshape(testdatadenormalized(i).face,50,50));
%     hold on;
%     plot(validation(1:68),validation(69:136),'r.','MarkerSize',20);
%     title('Ground Truth')
%     
%   
%     saveas(fig,char(strcat(root,'\resultsSAE\',num2str(i), '.jpg')))
% 
%     pause(5);
%     close all;
% end

%% show results on train set
load('landmark_prd.mat') ;
load('pose_prd.mat');
load('testdatadenormalized.mat');
root = pwd;
for i = 1: size(testdatadenormalized,2)
    validation = testdatadenormalized(i).groundtruth;
    fig = figure;
    subplot(1,2,1);
    imshow(reshape(testdatadenormalized(i).face,50,50));
    hold on;
    plot(landmark_prd(1:68,i),landmark_prd(69:136,i),'r.','MarkerSize',20);
    title('My Predictions')
    subplot(1,2,2);
    imshow(reshape(testdatadenormalized(i).face,50,50));
    hold on;
    plot(validation(1:68),validation(69:136),'r.','MarkerSize',20);
    title('Ground Truth')
    
  
    saveas(fig,char(strcat(root,'\resultsSAE\',num2str(i), '.jpg')))

    pause(5);
    close all;
end