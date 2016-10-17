function [normalizedim] = normalizePic(pic)
%     normalizedim = (pic(:) - min(pic(:))) / ( max(pic(:)) - min(pic(:)) );
%     normalizedim = reshape(normalizedim, size(pic, 1), size(pic,2));
reshaped_pic = reshape(pic,[],1);
reshaped_pic=reshaped_pic-mean(reshaped_pic(:));
reshaped_pic=reshaped_pic/std(reshaped_pic(:));
normalizedim = reshape(reshaped_pic,size(pic,1),size(pic,2));
end