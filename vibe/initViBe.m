%% Parameters
numberOfSamples         = param.numberOfSamples;
matchingThreshold       = param.matchingThreshold;
matchingNumber          = param.matchingNumber;
updateFactor            = param.updateFactor;
numberOfHistoryImages   = param.numberOfHistoryImages;

%% Initialize ViBe
% originalImage=vidFrame;%%开辟一块存储用于保存各像素点的原始背景模型
% flags=unit8(zeros(height, width));%%为每一个像素设置一个标志只是该像素点是否保存有原始背景模型，1为保存了原始背景模型，初始化为0
% counters=unit8(zeros(height,width));%%为每一个像素点设置一个计数器，用来统计该像素点连续被判为前景的帧数

historyImages = cell(1, numberOfHistoryImages);
for ii = 1:length(historyImages)
    historyImages{ii} = vidFrame;
end
%hsvImage=hsvFrame;
historyBuffer = cell(1, numberOfSamples - numberOfHistoryImages);
for ii = 1:length(historyBuffer)  %%  通过人为地对图像加入一定范围内的随机噪声构建样本集，这也时官网代码中的做法。
%     historyBuffer{ii} = vidFrame + double(floor(rand(height, width))*20 - 10);
    historyBuffer{ii} = vidFrame + double(floor(rand(height, width)*20- 10) );
end

%% Random Part
% 对于时间取样和空间邻域更新策略，在官网代码实现的时候确实很巧妙，
% 个人觉得很体现代码功力。实现的时候，如果对每一个点都进行概率判断，
% 就相当于遍历了每一个像素点，复杂度为O(height*width)；换个角度思考，
% 每个点都有一定概率（比如1/rate）才更新，等同于整幅图图像每次会更新
% 一定概率部分个点，即1/rate*height*width个点更新，这样整体的计算复
% 杂度就会降到O(1/rate*height*width)。
size =2*max(height, width) + 1; 
% jump[] from 1 to 2*updateFactor
% 官网在实现的时候是通过随机步长的方式，
% 比如，步长为1~(2*rate - 1)之间的一个随机数，
% 每次更新一个点就向前移动一定步长。（假设整幅
% 图像存储为一个一维数组，每次通过加步长计算索
% 引）这样计算下来，平均每次移动的步长约等于rate，
% 那么整幅图像就会有1/rate*height*width进行更新，等价实现
jump = floor(rand(1, size)*2*updateFactor) + 1;%%随机步长
% neighborX, Y represent the neighbor index
neighborX = floor(rand(1, size)*3) - 1;% -1~1
neighborY = floor(rand(1, size)*3) - 1;
% position[] from 1 to numberOfSamples
position = floor(rand(1, size)*numberOfSamples) + 1;  %1~20

disp('Initialize ViBe')