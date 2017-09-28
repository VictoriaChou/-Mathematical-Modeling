clc;
clear;
obj = VideoReader('input.avi');
totalFrames = obj.NumberOfFrames;
DIRIMG = 'F:\sm_imgs\';
DIRAVI = 'F:\sm_avis\';
delete(strcat(DIRIMG,'*.bmp'));
%cntFrames = 23;

for k = 1 : totalFrames
    frame = read(obj,k);
    imwrite(frame,strcat(DIRIMG, num2str(k),'.bmp'),'bmp');
end