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
%     ǰ���ָ�Ĺ��̼��ǶԵ�ǰ֡�������뱳���������е����ؽ��бȽϣ�
%     �жϵ�ǰ�����Ƿ���ϱ�������������������ıȽ��ֶξ��Ǽ�������
%     ֵ�뱳��ģ���е����������ľ��룬���ڵ�ͨ�����Ҷ�ͼ���Ǿ���ֵ��
%     ������ͨ������ɫͼ������ɫ�ռ���������ľ��룬���趨����ֵ����
%     �Ƚϣ������Ƿ�С����ֵ��Ȼ����ͳ��С����ֵ�ĵ����Ƿ��㹻��
%     �������Ƿ����������������������.�����Ŀ�㹻����������Ϊ��������֮����Ϊǰ����

segmentationMap = uint8(ones(height, width)*(matchingNumber - 1));
% First and Second history Image structure
distance1 = abs(buffer - historyImages{1}) <= matchingThreshold;
distance2 = abs(buffer - historyImages{2}) <= matchingThreshold;
distance3 = (hsvbuffer(:,:,1))./(hsvImage(:,:,1))<=param.beta2;
distance7 = (hsvbuffer(:,:,1))./(hsvImage(:,:,1))>=param.beta1;
distance4 = hsvbuffer(:,:,2)-hsvImage(:,:,2)<=param.sigma;
distance5 = hsvbuffer(:,:,3)-hsvImage(:,:,3)<=param.sigma;
% distance1��distance2��ƥ��ʱsegmentation��ii,jj��Ϊ0��ֻ��һ��ƥ��ʱΪ1������ƥ��ʱΪ2
for ii = 1:height
    for jj = 1:width
        if ~distance1(ii, jj)
            segmentationMap(ii, jj) = matchingNumber;
        end
        if distance2(ii, jj)
            segmentationMap(ii, jj) = segmentationMap(ii, jj) - 1;
        end
        if distance3(ii,jj)&&distance7(ii,jj)&& distance4(ii,jj)&& distance5(ii,jj)
            shadowMap(ii,jj)=0;%���Ϊ��Ӱ
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
segmentationMap=segmentationMap.*shadowMap;%���Ϊ��Ӱ����*0  �����뱳��
