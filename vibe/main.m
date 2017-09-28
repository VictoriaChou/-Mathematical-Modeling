clear; clc; close all;
%% Parameters
param.numberOfSamples           = 20; %%ÿ�����ص���������
param.matchingThreshold         = 20;  %%Sqthere�뾶
param.matchingNumber            = 2; %% ��ֵ��minָ��
param.updateFactor              = 16; %%�Ӳ�������
param.numberOfHistoryImages     = 2; %%
param.lastHistoryImageSwapped   = 0;
% param.ghosth=15;
% param.backth=20;
param.n=50;
param.step=5;
param.i=5;
param.t=5;
param.AreaThreshold=64;
param.haveTrained=false;%�Ƿ�ѵ�����
param.beta1=0.6;
param.beta2=0.99;
param.sigma=0.15;

%% Video Information
% filename = 'F:\������Ƶ\�ͷֱ��ʼ���Ƭ��\MAH00268 00_00_00-00_00_03.MP4';%��Ӱ
% filename = 'F:\������Ƶ\�ͷֱ��ʼ���Ƭ��\MAH00202 00_00_04-.MP4'%������ˤ�������Ӷ�����Ӱ��
% filename = 'F:\������Ƶ\�ͷֱ��ʼ���Ƭ��\ˤ��\ghosttest.mp4';
filename='C:\Users\Administrator\Documents\MATLAB\input.avi';
vidObj = VideoReader(filename);
nFrame = size(vidObj,2);
sequenceNum=1;%��ʾ�ڼ�֡
height = vidObj.Height;
width = vidObj.Width;

param.height = height;
param.width = width;
param.counter=1;
%���ĺ�����������
param.heart=cell(zeros(1,2));
RatioMap=zeros(1,width);
tic;
%% ViBe Moving Object Detection
    isGhost=false;
    vidFrame = VideoWriter(vidObj.name);
%     name2=['F:\������Ƶ\ѵ��Ƭ��\ԭ��Ƶ',num2str(param.counter),'.jpg']
%     imwrite(vidFrame,name2);
    %rgbFrame = rgb2gray(vidFrame);
    %hsvFrame=rgb2hsv(vidFrame);
    %vidFrame = double(rgbFrame);
    shadowMap=ones(height,width);%1��ʾ�õ㲻����Ӱ��0��ʾ�õ�����Ӱ
    shadowMap=double(shadowMap);
    if param.counter==1
        initViBe;
    end
    
    segmentationMap = vibeSegmentation(vidFrame,hsvFrame,shadowMap, historyImages,hsvImage, historyBuffer, param);
    [historyImages, historyBuffer] = vibeUpdate(vidFrame,shadowMap ,segmentationMap, historyImages, historyBuffer, param, ...
        jump, neighborX, neighborY, position);
    segmentationMap = medfilt2(segmentationMap);
    %���Ƚ�ͼ�������������ڽ���һ����ֵ�˲���Ȼ���ٽ�ͼ���С��ԭ�����Դﵽ���õ��˲�Ч��
    X1=imresize(segmentationMap,0.5);
    X2=medfilt2(X1);
    segmentationMap=imresize(X2,2);
    segmentationMap=im2bw(segmentationMap);
    figure(1),imshow(segmentationMap);
%     name=['F:\������Ƶ\ˤ����Ƶ\hsv_Vibe',num2str(param.counter),'.jpg'];
%     imwrite(segmentationMap,name);
%     name2=['F:\������Ƶ\ˤ����Ƶ\ԭͼ',num2str(param.counter),'.jpg'];
%     imwrite(rgbFrame,name2);

    
    
    
    
    
    
    
    

         
    param.counter=param.counter+1;

toc