function [normalizedim] = zero_one_normalization(pic)
%% This comment is for github
    normalizedim = (pic(:) - min(pic(:))) / ( max(pic(:)) - min(pic(:)) );
    normalizedim = reshape(normalizedim, size(pic, 1), size(pic,2));

end