function [rotated_image,rotated_gts] = rotate_image(image,gt,angle) 
% Rotate an image with any angle for data augmentation purposes
    if size(image,3) ~= 1
        image = rgb2gray(image); % gray
    end
    origin=size(image)/2+.5; % center of image
    rotated_image = imrotate(image,angle,'bilinear','crop ') ;
    rotated_gts  = rotate_points(gt,angle,[origin(1,2) ; origin(1,1)]);
end