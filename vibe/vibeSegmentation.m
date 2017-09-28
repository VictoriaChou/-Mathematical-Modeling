function [segmentationMap,shadowMap] = vibeSegmentation(buffer,hsvbuffer, shadowMap,historyImages,hsvImage, historyBuffer, param)
%% Parameters
height  = param.height;
width   = param.width;
numberOfSamples         = param.numberOfSamples;
matchingThreshold       = param.matchingThreshold;
%     matchingThreshold =myotsu(buffer);
matchingNumber          = param.matchingNumber;
numberOfHistoryImages   = param.numberOfHistoryImages;

%% Segmentation
%     前景分割的过程即是对当前帧的像素与背景样本集中的像素进行比较，
%     判断当前像素是否符合背景样本的特征。具体的比较手段就是计算像素
%     值与背景模型中的所有样本的距离，对于单通道即灰度图就是绝对值，
%     对于三通道即彩色图就是颜色空间中两个点的距离，与设定的阈值进行
%     比较，计算是否小于阈值；然后，再统计小于阈值的点数是否足够多
%     （比如是否多于两个样本符合条件）.如果数目足够，则新像素为背景；反之，则为前景。

segmentationMap = uint8(ones(height, width)*(matchingNumber - 1));
% First and Second history Image structure
distance1 = abs(buffer - historyImages{1}) <= matchingThreshold;
distance2 = abs(buffer - historyImages{2}) <= matchingThreshold;
distance3 = (hsvbuffer(:,:,1))./(hsvImage(:,:,1))<=param.beta2;
distance7 = (hsvbuffer(:,:,1))./(hsvImage(:,:,1))>=param.beta1;
distance4 = hsvbuffer(:,:,2)-hsvImage(:,:,2)<=param.sigma;
distance5 = hsvbuffer(:,:,3)-hsvImage(:,:,3)<=param.sigma;
% distance1和distance2均匹配时segmentation（ii,jj）为0，只有一个匹配时为1，均不匹配时为2
for ii = 1:height
    for jj = 1:width
        if ~distance1(ii, jj)
            segmentationMap(ii, jj) = matchingNumber;
        end
        if distance2(ii, jj)
            segmentationMap(ii, jj) = segmentationMap(ii, jj) - 1;
        end
        if distance3(ii,jj)&&distance7(ii,jj)&& distance4(ii,jj)&& distance5(ii,jj)
            shadowMap(ii,jj)=0;%检测为阴影
        end
    end
end
% match the image and samples
numberOfTests = numberOfSamples - numberOfHistoryImages;
for kk = 1:numberOfTests
    distance6 = uint8(abs(buffer - historyBuffer{kk}) <= matchingThreshold);
    segmentationMap = segmentationMap - distance6;
end

segmentationMap = uint8(segmentationMap*255);
shadowMap=uint8(shadowMap);
segmentationMap=segmentationMap.*shadowMap;%检测为阴影部分*0  被划入背景
