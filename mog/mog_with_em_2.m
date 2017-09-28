clc;
clear;
obj = VideoReader('input.avi');
totalFrames = obj.NumberOfFrames;
DIRIMG = 'C:\Users\Administrator\Documents\MATLAB\mog\imgs\';
DIRAVI = 'C:\Users\Administrator\Documents\MATLAB\mog\avi\';
delete(strcat(DIRIMG,'*.bmp'));
%cntFrames = 23;
%��ȡ��Ƶ֡
for k = 1 : totalFrames
    frame = read(obj,k);
    imwrite(frame,strcat(DIRIMG, num2str(k),'.bmp'),'bmp');
end
% ��ȡÿһ֡������ֵ
T = 2.5; %ƫ����ֵ
alpha = 0.005; %ѧϰ��
thresh = 0.3; %ǰ����ֵ
sd_init = 6; % ��ʼ����׼��
d = 1;
C = 3;
K = 3;
I = imread(strcat(DIRIMG, '1.bmp'));
fr_bw = im2double(rgb2gray(I));
[row, col] = size(fr_bw);
fg_pre = zeros(row, col);  % ǰ������
bg = zeros(row, col);  % ��������
w = zeros(row, col, C);
mu = zeros(row, col, C);
sigma = zeros(row, col, C);
X = zeros(totalFrames,row*col);
u_diff = zeros(row, col, C); % ������ĳ����˹ģ�;�ֵ�ľ��Ծ���
p = alpha/(1/C); % ��ʼ��p�������������¾�ֵ�ͱ�׼��
rank = zeros(1,C); % ������˹�ֲ������ȼ�
pixel_depth = 8; % ÿ������8bit
pixel_range = 2^pixel_depth - 1; %����ֵ��Χ[0,255]

for r=1:row
    for c=1:col  
        for it=1:totalFrames
            I1 = imread(strcat(DIRIMG, num2str(it),'.bmp'));
            fr_bw = rgb2gray(I1);
            X(it) = fr_bw(r,c);
        end
        [label, model, llh] = mixGaussEm(X,C);
        for co = 1:C
            w(r,c,co) = model.w(co);
            mu(r,c,co) = model.mu(co);
            sigma(r,c,co) = model.Sigma(co);
        end
    end
end

 for n=1:totalFrames
     frame = strcat(DIRIMG, num2str(n), '.bmp');
     img = imread(frame);
     grayimg = double(rgb2gray(I1));
     
%     %fr_bw2 = im2double(I1);
%     % �������������m����˹ģ�;�ֵ�ľ��Ծ���
     for m=1:C
         u_diff(:,:,m) = abs(double(img(:,:,m)) - mu(:,:,m));
     end
     for i=1:row
         for j=1:col
             flag = 0;
             for k=1:C
%  ����fuzzyģ��
                muup(i,j,k) = mu(i,j,k) + K*(sigma(i,j,k))^2;
                mulow(i,j,k) = mu(i,j,k) - K*(sigma(i,j,k))^2;
                 if(fr_bw(i,j) < mulow(i,j,k))
                     mu(i,j,k) = mulow(i,j,k);
                 elseif(fr_bw(i,j) > muup(i,j,k))
                    mu(i,j,k) = muup(i,j,k);
                else
                    mu(i,j,k) = mu(i,j,k);
                end
                if(abs(u_diff(i,j,k)) <= T*sigma(i,j,k))
                    flag = 1;
                    %����Ȩ�ء���ֵ����׼�p
                    w(i,j,k) = (1-alpha)*w(i,j,k) + alpha*flag;
                     p = alpha/w(i,j,k);
                     mu(i,j,k) = (1-p)*mu(i,j,k) + p*double(img(i,j));
                     sigma(i,j,k) = sqrt((1-p)*(sigma(i,j,k)^2) + p*(double(img(i,j)) - mu(i,j,k))^2);
                 else
                     w(i,j,k) = (1 - alpha)*w(i,j,k);
                 end
             end
             bg(i,j) = 0;
             for k=1:C
                 bg(i,j) = bg(i,j) + mu(i,j,k)*w(i,j,k);
             end
            % û��������ֵƥ���ģ�ͣ��򴴽��µ�ģ��
            if(flag == 0)
                [min_w, min_w_index] = min(w(i,j,:)); % Ѱ����СȨ��ֵ
                mu(i,j,min_w_index) = fr_bw(i,j);
                sigma(i,j,min_w_index) = sd_init;
            end
            rank = w(i,j,:)./sigma(i,j,:); % ����ģ�����ȼ�
            rank_ind = [1:1:C];
            % ����ǰ��
            fg_pre(i,j) = 0;
            while((flag == 0) && (k <= C))
                if(abs(u_diff(i,j,rank_ind(k))) <= T*sigma(i,j,rank_ind(k))) %�������k����˹ģ��ƥ��
                    fg_pre(i,j) = 0;
                else
                    fg_pre(i,j) = 255;
                end
                k = k+1;
            end
        end
    end
    % ��ͼ�������̬ѧ����
    SE1 = strel('disk', 1);
    SE2 = strel('disk', 4);
    %fg = imerode(fg_pre, SE1);
    fg = imdilate(fg_pre, SE1);
    fg = imfill(fg, 'holes');
    fg = imclearborder(fg);
    
    %dFg = imdilate(eFg,SE2);
    %closeFg = imclose(fg_pre, SE2);
    %openFg = imopen(closeFg, SE1);
    imwrite(fg_pre,...
        strcat(DIRIMG, 'fg',...
        num2str(n),'.bmp'),'bmp');
    file=dir(strcat(DIRIMG,'fg',num2str(n),'.bmp'));  
end


obj_gray = VideoWriter(strcat(DIRAVI,'fg.avi'));   %��ת���ɵ���Ƶ����

%������ͼƬ����avi�ļ�
open(obj_gray);
for k = 1: totalFrames
    fname = strcat(DIRIMG,'fg',num2str(k),'.bmp');
    frame = imread(fname);
    writeVideo(obj_gray, frame);
end
close(obj_gray);
