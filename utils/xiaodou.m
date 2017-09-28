%% Video Stabilization Using Point Feature Matching
% This example shows how to stabilize a video that was captured from a
% jittery platform. One way to stabilize a video is to track a salient
% feature in the image and use this as an anchor point to cancel out all
% perturbations relative to it. This procedure, however, must be
% bootstrapped with knowledge of where such a salient feature lies in the
% first video frame. In this example, we explore a method of video
% stabilization that works without any such _a priori_ knowledge. It
% instead automatically searches for the "background plane" in a video
% sequence, and uses its observed distortion to correct for camera motion.
%
% This stabilization algorithm involves two steps. First, we determine the
% affine image transformations between all neighboring frames of a video
% sequence using the |estimateGeometricTransform| function applied to point
% correspondences between two images. Second, we warp the video frames to
% achieve a stabilized video. We will use the Computer Vision System
% Toolbox(TM), both for the algorithm and for display.
%
% This example is similar to the <videostabilize.html Video Stabilization
% Example>. The main difference is that the Video Stabilization Example is
% given a region to track while this example is given no such knowledge.
% Both examples use the same video.

%  Copyright 2009-2010 The MathWorks, Inc.

%% Step 1. Read Frames from a Movie File
% Here we read in the first two frames of a video sequence. We read them as
% intensity images since color is not necessary for the stabilization
% algorithm, and because using grayscale images improves speed. Below we
% show both frames side by side, and we produce a red-cyan color composite
% to illustrate the pixel-wise difference between them. There is obviously
% a large vertical and horizontal offset between the two frames.
filename = 'input.avi';
hVideoSrc = vision.VideoFileReader(filename, 'ImageColorSpace', 'Intensity');

imgA = step(hVideoSrc); % Read first frame into imgA

imgB = step(hVideoSrc); % Read second frame into imgB

%figure; imshowpair(imgA, imgB, 'montage');
%title(['Frame A', repmat(' ',[1 70]), 'Frame B']);

%%
%figure; imshowpair(imgA,imgB,'ColorChannels','red-cyan');
%title('Color composite (frame A = red, frame B = cyan)');

%% Step 2. Collect Salient Points from Each Frame
% Our goal is to determine a transformation that will correct for the
% distortion between the two frames. We can use the
% |estimateGeometricTransform| function for this, which will return an
% affine transform. As input we must provide this function with a set of
% point correspondences between the two frames. To generate these
% correspondences, we first collect points of interest from both frames,
% then select likely correspondences between them.
%
% In this step we produce these candidate points for each frame. To have
% the best chance that these points will have corresponding points in the
% other frame, we want points around salient image features such as
% corners. For this we use the |detectFASTFeatures| function, which
% implements one of the fastest corner detection algorithms.
%
% The detected points from both frames are shown in the figure below.
% Observe how many of them cover the same image features, such as points
% along the tree line, the corners of the large road sign, and the corners
% of the cars.
ptThresh = 0.1;
pointsA = detectFASTFeatures(imgA, 'MinContrast', ptThresh);
pointsB = detectFASTFeatures(imgB, 'MinContrast', ptThresh);

% Display corners found in images A and B.
% figure; imshow(imgA); hold on;
% plot(pointsA);
% title('Corners in A');
% 
% figure; imshow(imgB); hold on;
% plot(pointsB);
% title('Corners in B');

%% Step 3. Select Correspondences Between Points
% Next we pick correspondences between the points derived above. For each
% point, we extract a Fast Retina Keypoint (FREAK) descriptor centered
% around it. The matching cost we use between points is the Hamming
% distance since FREAK descriptors are binary. Points in frame A and frame
% B are matched putatively. Note that there is no uniqueness constraint, so
% points from frame B can correspond to multiple points in frame A.

% Extract FREAK descriptors for the corners
[featuresA, pointsA] = extractFeatures(imgA, pointsA);
[featuresB, pointsB] = extractFeatures(imgB, pointsB);

%%
% Match features which were found in the current and the previous frames.
% Since the FREAK descriptors are binary, the |matchFeatures| function 
% uses the Hamming distance to find the corresponding points.
indexPairs = matchFeatures(featuresA, featuresB);
pointsA = pointsA(indexPairs(:, 1), :);
pointsB = pointsB(indexPairs(:, 2), :);

%%
% The image below shows the same color composite given above, but added are
% the points from frame A in red, and the points from frame B in green.
% Yellow lines are drawn between points to show the correspondences
% selected by the above procedure. Many of these correspondences are
% correct, but there is also a significant number of outliers.

%figure; showMatchedFeatures(imgA, imgB, pointsA, pointsB);
%legend('A', 'B');

%% Step 4. Estimating Transform from Noisy Correspondences
% Many of the point correspondences obtained in the previous step are
% incorrect. But we can still derive a robust estimate of the geometric
% transform between the two images using the M-estimator SAmple Consensus
% (MSAC) algorithm, which is a variant of the RANSAC algorithm. The MSAC
% algorithm is implemented in the |estimateGeometricTransform| function.
% This function, when given a set of point correspondences, will search for
% the valid inlier correspondences. From these it will then derive the
% affine transform that makes the inliers from the first set of points
% match most closely with the inliers from the second set. This affine
% transform will be a 3-by-3 matrix of the form:
%
%  [a_1 a_3 0;
%   a_2 a_4 0;
%   t_x t_y 1]
%
% The parameters $a$ define scale, rotation, and sheering effects of the
% transform, while the parameters $t$ are translation parameters. This
% transform can be used to warp the images such that their corresponding
% features will be moved to the same image location.
%
% A limitation of the affine transform is that it can only alter the
% imaging plane. Thus it is ill-suited to finding the general distortion
% between two frames taken of a 3-D scene, such as with this video taken
% from a moving car. But it does work under certain conditions that we
% shall describe shortly.

[tform, pointsBm, pointsAm] = estimateGeometricTransform(...
    pointsB, pointsA, 'affine');
imgBp = imwarp(imgB, tform, 'OutputView', imref2d(size(imgB)));
pointsBmp = transformPointsForward(tform, pointsBm.Location);

%%
% Below is a color composite showing frame A overlaid with the reprojected
% frame B, along with the reprojected point correspondences. The results
% are excellent, with the inlier correspondences nearly exactly coincident.
% The cores of the images are both well aligned, such that the red-cyan
% color composite becomes almost purely black-and-white in that region.
%
% Note how the inlier correspondences are all in the background of the
% image, not in the foreground, which itself is not aligned. This is
% because the background features are distant enough that they behave as if
% they were on an infinitely distant plane. Thus, even though the affine
% transform is limited to altering only the imaging plane, here that is
% sufficient to align the background planes of both images. Furthermore, if
% we assume that the background plane has not moved or changed
% significantly between frames, then this transform is actually capturing
% the camera motion. Therefore correcting for this will stabilize the
% video. This condition will hold as long as the motion of the camera
% between frames is small enough, or, conversely, if the sample time of the
% video is high enough.

% figure;
% showMatchedFeatures(imgA, imgBp, pointsAm, pointsBmp);
% legend('A', 'B');

%% Step 5. Transform Approximation and Smoothing
% Given a set of video frames $T_{i}, \quad i=0,1,2 \ldots$, we can now use
% the above procedure to estimate the distortion between all frames $T_i$
% and $T_{i+1}$ as affine transforms, $H_i$. Thus the cumulative distortion
% of a frame $i$ relative to the first frame will be the product of all the
% preceding inter-frame transforms, or
%
% $H_{cumulative,i} = H_i \prod_{j=0}^{i-1}$
%
% We could use all the six parameters of the affine transform above, but,
% for numerical simplicity and stability, we choose to re-fit the matrix as
% a simpler scale-rotation-translation transform. This has only four free
% parameters compared to the full affine transform's six: one scale factor,
% one angle, and two translations. This new transform matrix is of the
% form:
%
%  [s*cos(ang)  s*-sin(ang)  0;
%   s*sin(ang)   s*cos(ang)  0;
%          t_x         t_y   1]
%
% We show this conversion procedure below by fitting the above-obtained
% transform $H$ with a scale-rotation-translation equivalent, $H_{sRt}$. To
% show that the error of converting the transform is minimal, we reproject
% frame B with both transforms and show the two images below as a red-cyan
% color composite. As the image appears black and white, obviously the
% pixel-wise difference between the different reprojections is negligible.

% Extract scale and rotation part sub-matrix.
H = tform.T;
R = H(1:2,1:2);
% Compute theta from mean of two possible arctangents
theta = mean([atan2(R(2),R(1)) atan2(-R(3),R(4))]);
% Compute scale from mean of two stable mean calculations
scale = mean(R([1 4])/cos(theta));
% Translation remains the same:
translation = H(3, 1:2);
% Reconstitute new s-R-t transform:
HsRt = [[scale*[cos(theta) -sin(theta); sin(theta) cos(theta)]; ...
  translation], [0 0 1]'];
tformsRT = affine2d(HsRt);

imgBold = imwarp(imgB, tform, 'OutputView', imref2d(size(imgB)));
imgBsRt = imwarp(imgB, tformsRT, 'OutputView', imref2d(size(imgB)));

% figure(2), clf;
% imshowpair(imgBold,imgBsRt,'ColorChannels','red-cyan'), axis image;
% title('Color composite of affine and s-R-t transform outputs');

%% Step 6. Run on the Full Video
% Now we apply the above steps to smooth a video sequence. For readability,
% the above procedure of estimating the transform between two images has
% been placed in the MATLAB(R) function
% <matlab:edit(fullfile(matlabroot,'toolbox','vision','visiondemos','cvexEstStabilizationTform.m')) |cvexEstStabilizationTform|>.
% The function
% <matlab:edit(fullfile(matlabroot,'toolbox','vision','visiondemos','cvexTformToSRT.m')) |cvexTformToSRT|>
% also converts a general affine transform into a
% scale-rotation-translation transform.
%
% At each step we calculate the transform $H$ between the present frames.
% We fit this as an s-R-t transform, $H_{sRt}$. Then we combine this the
% cumulative transform, $H_{cumulative}$, which describes all camera motion
% since the first frame. The last two frames of the smoothed video are
% shown in a Video Player as a red-cyan composite.
%
% With this code, you can also take out the early exit condition to make
% the loop process the entire video.

% Reset the video source to the beginning of the file.
reset(hVideoSrc);
                      
hVPlayer = vision.VideoPlayer; % Create video viewer

% Process all frames in the video
movMean = step(hVideoSrc);
imgB = movMean;
imgBp = imgB;
correctedMean = imgBp;
ii = 2;
Hcumulative = eye(3);
while ~isDone(hVideoSrc) && ii < 10
    % Read in new frame
    imgA = imgB; % z^-1
    imgAp = imgBp; % z^-1
    imgB = step(hVideoSrc);
    movMean = movMean + imgB;

    % Estimate transform from frame A to frame B, and fit as an s-R-t
    H = cvexEstStabilizationTform(imgA,imgB);
    HsRt = cvexTformToSRT(H);
    Hcumulative = HsRt * Hcumulative;
    imgBp = imwarp(imgB,affine2d(Hcumulative),'OutputView',imref2d(size(imgB)));

    % Display as color composite with last corrected frame
    step(hVPlayer, imfuse(imgAp,imgBp,'ColorChannels','red-cyan'));
    correctedMean = correctedMean + imgBp;
    
    ii = ii+1;
end
correctedMean = correctedMean/(ii-2);
movMean = movMean/(ii-2);

% Here you call the release method on the objects to close any open files
% and release memory.
release(hVideoSrc);
release(hVPlayer);

obj = VideoReader(hVideoSrc);
totalFrames = obj.NumberOfFrames;
DIRIMG = 'F:\sm_imgs_nofuzzy\';
DIRAVI = 'F:\sm_avis_nofuzzy\';
delete(strcat(DIRIMG,'*.bmp'));
%cntFrames = 23;
%获取视频帧
for k = 1 : totalFrames
    frame = read(obj,k);
    imwrite(frame,strcat(DIRIMG, num2str(k),'.bmp'),'bmp');
end
% 获取每一帧的向量值
T = 2.5; %偏差阈值
alpha = 0.005; %学习率
thresh = 0.3; %前景阈值
sd_init = 6; % 初始化标准差
d = 1;
C = 3;
I = imread(strcat(DIRIMG, '1.bmp'));
fr_bw = im2double(rgb2gray(I));
[row, col] = size(fr_bw);
fg_pre = zeros(row, col);  % 前景矩阵
bg = zeros(row, col);  % 背景矩阵
w = zeros(row, col, C);
mu = zeros(row, col, C);
sigma = zeros(row, col, C);
X = zeros(totalFrames,row*col);
for it=1:totalFrames
    I1 = imread(strcat(DIRIMG, num2str(it),'.bmp'));
    fr_bw = im2double(rgb2gray(I1));
    X1(it,:)=reshape(fr_bw, 1, row*col);
end
x = reshape(X1, 1, totalFrames*row*col);
[label, model, llh] = mixGaussEm(x,C);
for k=1:C
    w(:,:,k) = model.w(k)
    mu(:,:,k) = model.mu(k);
    Sigma = reshape(model.Sigma,1,3);
    sigma(:,:,k) = Sigma(k);
end 

%n = 10^6;
%[X,label,model] = mixGaussRnd(d,C,n); 
% w = zeros(row, col, C);
% mu = zeros(row, col, C);
% sigma = zeros(row, col, C);
% u_diff = zeros(row, col, C); % 像素与某个高斯模型均值的绝对距离
% p = alpha/(1/C); % 初始化p向量，用来更新均值和标准差
% rank = zeros(1,C); % 各个高斯分布的优先级
% pixel_depth = 8; % 每个像素8bit
% pixel_range = 2^pixel_depth - 1; %像素值范围[0,255]
% 
% % for k=1:C
% %     w(:,:,k) = model.w(k)
% %     mu(:,:,k) = model.mu(k);
% %     Sigma = reshape(model.Sigma,1,3);
% %     sigma(:,:,k) = Sigma(k);
% % end
% 
for n=1:totalFrames
    frame = strcat(DIRIMG, num2str(n), '.bmp');
    I1 = imread(frame);
    fr_bw = I1;
    % 计算新像素与第m个高斯模型均值的绝对距离
    for m=1:C
        u_diff(:,:,m) = abs(double(fr_bw(:,:,m)) - double(mu(:,:,m)));
    end
    for i=1:row
        for j=1:col
            flag = 0;
            for k=1:C
                if(abs(u_diff(i,j,k)) <= T*sigma(i,j,k))
                    flag = 1;
                    %更新权重、均值、标准差、p
                    w(i,j,k) = (1-alpha)*w(i,j,k) + alpha*flag;
                    p = alpha/w(i,j,k);
                    mu(i,j,k) = (1-p)*mu(i,j,k) + p*double(fr_bw(i,j));
                    sigma(i,j,k) = sqrt((1-p)*(sigma(i,j,k)^2) + p*((double(fr_bw(i,j)) - mu(i,j,k)))^2);
                else
                    w(i,j,k) = (1 - alpha)*w(i,j,k);
                end
            end
            bg(i,j) = 0;
            for k=1:C
                bg(i,j) = bg(i,j) + mu(i,j,k)*w(i,j,k);
            end
            % 没有与像素值匹配的模型，则创建新的模型
            if(flag == 0)
                [min_w, min_w_index] = min(w(i,j,:)); % 寻找最小权重值
                mu(i,j,min_w_index) = double(fr_bw(i,j));
                sigma(i,j,min_w_index) = sd_init;
            end
            rank = w(i,j,:)./sigma(i,j,:); % 计算模型优先级
            rank_ind = [1:1:C];
            % 计算前景
            fg_pre(i,j) = 0;
            while((flag == 0) && (k <= C))
                if(abs(u_diff(i,j,rank_ind(k))) <= T*sigma(i,j,rank_ind(k))) %像素与第k个高斯模型匹配
                    fg_pre(i,j) = 0;
                else
                    fg_pre(i,j) = 255;
                end
                k = k+1;
            end
        end
    end
    % 对图像进行形态学处理
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


obj_gray = VideoWriter(strcat(DIRAVI,'fg.avi'));   %所转换成的视频名称

%将单张图片存在avi文件
open(obj_gray);
for k = 1: totalFrames
    fname = strcat(DIRIMG,'fg',num2str(k),'.bmp');
    frame = imread(fname);
    writeVideo(obj_gray, frame);
end
close(obj_gray);

%%
% During computation, we computed the mean of the raw video frames and of
% the corrected frames. These mean values are shown side-by-side below. The
% left image shows the mean of the raw input frames, proving that there was
% a great deal of distortion in the original video. The mean of the
% corrected frames on the right, however, shows the image core with almost
% no distortion. While foreground details have been blurred (as a necessary
% result of the car's forward motion), this shows the efficacy of the
% stabilization algorithm.

%figure; imshowpair(movMean, correctedMean, 'montage');

title(['Raw input mean', repmat(' ',[1 50]), 'Corrected sequence mean']);

%% References
% [1] Tordoff, B; Murray, DW. "Guided sampling and consensus for motion
% estimation." European Conference n Computer Vision, 2002.
%
% [2] Lee, KY; Chuang, YY; Chen, BY; Ouhyoung, M. "Video Stabilization
% using Robust Feature Trajectories." National Taiwan University, 2009.
% 
% [3] Litvin, A; Konrad, J; Karl, WC. "Probabilistic video stabilization
% using Kalman filtering and mosaicking." IS&T/SPIE Symposium on Electronic
% Imaging, Image and Video Communications and Proc., 2003.
%
% [4] Matsushita, Y; Ofek, E; Tang, X; Shum, HY. "Full-frame Video
% Stabilization." Microsoft(R) Research Asia. CVPR 2005.

displayEndOfDemoMessage(mfilename)
