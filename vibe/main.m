clear; clc; close all;
%% Parameters
param.numberOfSamples           = 20; %%每个像素点样本个数
param.matchingThreshold         = 20;  %%Sqthere半径
param.matchingNumber            = 2; %% 阈值，min指数
param.updateFactor              = 16; %%子采样概率
param.numberOfHistoryImages     = 2; %%
param.lastHistoryImageSwapped   = 0;
% param.ghosth=15;
% param.backth=20;
param.n=50;
param.step=5;
param.i=5;
param.t=5;
param.AreaThreshold=64;
param.haveTrained=false;%是否训练完成
param.beta1=0.6;
param.beta2=0.99;
param.sigma=0.15;

%% Video Information
% filename = 'F:\自拍视频\低分辨率剪辑片段\MAH00268 00_00_00-00_00_03.MP4';%鬼影
% filename = 'F:\自拍视频\低分辨率剪辑片段\MAH00202 00_00_04-.MP4'%姜福义摔倒（垫子动、鬼影）
% filename = 'F:\自拍视频\低分辨率剪辑片段\摔倒\ghosttest.mp4';
filename='C:\Users\Administrator\Documents\MATLAB\input.avi';
vidObj = VideoReader(filename);
nFrame = size(vidObj,2);
sequenceNum=1;%表示第几帧
height = vidObj.Height;
width = vidObj.Width;

param.height = height;
param.width = width;
param.counter=1;
%质心横向及纵向坐标
param.heart=cell(zeros(1,2));
RatioMap=zeros(1,width);
tic;
%% ViBe Moving Object Detection
    isGhost=false;
    vidFrame = VideoWriter(vidObj.name);
%     name2=['F:\自拍视频\训练片段\原视频',num2str(param.counter),'.jpg']
%     imwrite(vidFrame,name2);
    %rgbFrame = rgb2gray(vidFrame);
    %hsvFrame=rgb2hsv(vidFrame);
    %vidFrame = double(rgbFrame);
    shadowMap=ones(height,width);%1表示该点不是阴影，0表示该点是阴影
    shadowMap=double(shadowMap);
    if param.counter==1
        initViBe;
    end
    
    segmentationMap = vibeSegmentation(vidFrame,hsvFrame,shadowMap, historyImages,hsvImage, historyBuffer, param);
    [historyImages, historyBuffer] = vibeUpdate(vidFrame,shadowMap ,segmentationMap, historyImages, historyBuffer, param, ...
        jump, neighborX, neighborY, position);
    segmentationMap = medfilt2(segmentationMap);
    %首先将图像扩大两倍，在进行一次中值滤波，然后再将图像大小还原，可以达到更好的滤波效果
    X1=imresize(segmentationMap,0.5);
    X2=medfilt2(X1);
    segmentationMap=imresize(X2,2);
    segmentationMap=im2bw(segmentationMap);
    figure(1),imshow(segmentationMap);
%     name=['F:\自拍视频\摔倒视频\hsv_Vibe',num2str(param.counter),'.jpg'];
%     imwrite(segmentationMap,name);
%     name2=['F:\自拍视频\摔倒视频\原图',num2str(param.counter),'.jpg'];
%     imwrite(rgbFrame,name2);

    
    
    
    
    
    
    
    

         
    param.counter=param.counter+1;

toc