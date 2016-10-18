function [translated_image,translated_gt] = translate_image(image,gt,drift_arr) 

translated_image = imtranslate(image,drift_arr);
translated_gt = [gt(:,1)+drift_arr(1) gt(:,2)+drift_arr(2)];
end
