close all;
clear;clc;
%% Read the hazy-free Image; 
inputImagePath_1 = "C:\Users\rosha\Downloads\WhatsApp Image 2023-10-10 at 15.46.48_3754045d.jpg";
HazefreeImg = imread(inputImagePath_1);
figure, imshow(HazefreeImg);
%% Read the depth-map Image; 
inputImagePath_2 = "C:\Users\rosha\Downloads\refineDepthMap.png";
Depth_map = imread(inputImagePath_2);
figure, imshow(Depth_map);
%% transmission-map Image; 
beta=0.5;
T = exp(-beta*double(Depth_map)/255);
figure, imshow(T, []);
% T = double(imread("C:\Users\rosha\Downloads\tr_house.png"));
title('transmission_map')
%%  global airlight;  
A = [255, 255,255];% R,G,B=255 corresponding to white air-light 

%% Haze Image formation; 
J =Atmospheric_Scattering(HazefreeImg, A, T);
figure, imshow(double(J)./255,[]);
%% Add rain streaks
streak_root = 'data/Streaks_Garg06/';
num_of_strtype = 5;
%streak = imread("C:\Users\rosha\Downloads\rainmap-crop (1).jpg");
%streak = imread("processed_image.jpg")
% original_streak = imread("C:\Users\rosha\Downloads\processed_image.jpg");
original_streak = imread("C:\Users\rosha\Downloads\rainmap-crop (1).jpg");
[w h d] = size(original_streak)
if d == 3
    resized_original_streak = im2gray(imresize(original_streak,[size(J,1),size(J,2)]));
else
    resized_original_streak = imresize(original_streak,[size(J,1),size(J,2)]);
end
h_flip_streak = imread("C:\Users\rosha\Downloads\flipped_image.jpg");
resized_h_flip_streak = imresize(h_flip_streak,[size(J,1),size(J,2)]);
gaussian_noise = wgn(size(J,1),size(J,2),5)
% final = uint8(J) + streak.*uint8(T) + streak.*uint8(1-T);
% imshow(final)
%% JT+sum(RT)
% rain_map = cat(3,resized_h_flip_streak,resized_original_streak,gaussian_noise)
final = [uint8(J) + uint8(resized_h_flip_streak) + uint8(resized_original_streak)].*uint8(T) + uint8((1-T)).*resized_original_streak + uint8((1-T)).*resized_h_flip_streak;
imshow(final)
