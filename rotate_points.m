function [rotated_landmarks] = rotate_points(points,theta,origin)
theta = -theta*(pi/180);  
% define a 60 degree counter-clockwise rotation matrix
R = [cos(theta) -sin(theta); sin(theta) cos(theta)];
center = double(repmat(origin, 1, length(points)));
v = double(points') - (center);

% do the rotation...
so = R*v;           % apply the rotation about the origin
vo = so + center;
% this can be done in one line as:
% vo = R*(v - center) + center
% pick out the vectors of rotated x- and y-data
rotated_landmarks = [vo(1,:);vo(2,:)]';
