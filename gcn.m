% GlOBAL CONTRAST NORMALIZATION
function [renrmlzd,gs] = gcn(images)
% load('images_denormalized.mat'); script kullanýrken aç
% input---------------------
% Size : 2500*4800, ilk boyut pixel say?s? ikinci boyut datasetteki imge say?s?
% Gray scaled 0-255 yo?unluk de?erli matris
%% Step 1: Local Mean Removal :
% For each image, subtract the local mean of all pixel values from the image:
lmr = images-repmat(mean(images),[size(images(:,1)) 1]) ;

%% Step 2: Image Norm Setting
for j = 1:size(images,2)
sos = sumsqr(lmr(:,j)); % sum of squares
srosos = sqrt(sos);% square root of
ins(:,j) = lmr(:,j)*100/srosos;
end

%% Step 3 : Global Mean Removal
global_mean = mean(ins,2); % find the global mean of each pixel in dataset
gmr = ins-repmat(global_mean,[1 size(images,2) ]);

%% Step 4 : Global Standardisation : Mean 0, std 1 Normalization
global_std  = (std(gmr'))';
gs = gmr./repmat(global_std,[1 size(images,2) ]);

%% Step 5 : Re-Normalize Images scaled between 0-255
renrmlzd= (gs-repmat(min(gs),[size(images,1) 1 ])) ./repmat((max(gs)-min(gs)),[size(images,1) 1])*255;

end