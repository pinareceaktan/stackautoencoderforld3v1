function [data_mat,landmarks_mat,labels_mat] = get_data_set_prepared(data_to_be_parsed,pose_partition)
imsize       = 50*50;
landmarksize = 68*2;
samplesize   = size(data_to_be_parsed,2);
data_mat = zeros(imsize,samplesize);
landmarks_mat  = zeros(landmarksize,samplesize);
labels_mat     = zeros(1,samplesize);
if ~pose_partition
for i = 1:size(data_to_be_parsed,2)
    try
    data_mat(:,i) = data_to_be_parsed(i).face;
    landmarks_mat(:,i)  = data_to_be_parsed(i).groundtruth;
    labels_mat(:,i) = data_to_be_parsed(i).pose;
%     labels_mat(:,i)     = data(i).pose;
    catch ME
        disp(i);
        disp(ME.identifier);
    end
end
else
    prompt  = 'Which pose do you want to fetch ? ';
    pose    = input(prompt);
%     [data_to_be_parsed.pose].'
    indices = find([data_to_be_parsed.pose] == pose(1,1));
    data_mat = double([data_to_be_parsed(indices).face]);
    landmarks_mat = [data_to_be_parsed(indices).groundtruth];
    labels_mat = [data_to_be_parsed(indices).pose];
end
