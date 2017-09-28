%% Parameters
numberOfSamples         = param.numberOfSamples;
matchingThreshold       = param.matchingThreshold;
matchingNumber          = param.matchingNumber;
updateFactor            = param.updateFactor;
numberOfHistoryImages   = param.numberOfHistoryImages;

%% Initialize ViBe
% originalImage=vidFrame;%%����һ��洢���ڱ�������ص��ԭʼ����ģ��
% flags=unit8(zeros(height, width));%%Ϊÿһ����������һ����־ֻ�Ǹ����ص��Ƿ񱣴���ԭʼ����ģ�ͣ�1Ϊ������ԭʼ����ģ�ͣ���ʼ��Ϊ0
% counters=unit8(zeros(height,width));%%Ϊÿһ�����ص�����һ��������������ͳ�Ƹ����ص���������Ϊǰ����֡��

historyImages = cell(1, numberOfHistoryImages);
for ii = 1:length(historyImages)
    historyImages{ii} = vidFrame;
end
%hsvImage=hsvFrame;
historyBuffer = cell(1, numberOfSamples - numberOfHistoryImages);
for ii = 1:length(historyBuffer)  %%  ͨ����Ϊ�ض�ͼ�����һ����Χ�ڵ����������������������Ҳʱ���������е�������
%     historyBuffer{ii} = vidFrame + double(floor(rand(height, width))*20 - 10);
    historyBuffer{ii} = vidFrame + double(floor(rand(height, width)*20- 10) );
end

%% Random Part
% ����ʱ��ȡ���Ϳռ�������²��ԣ��ڹ�������ʵ�ֵ�ʱ��ȷʵ�����
% ���˾��ú����ִ��빦����ʵ�ֵ�ʱ�������ÿһ���㶼���и����жϣ�
% ���൱�ڱ�����ÿһ�����ص㣬���Ӷ�ΪO(height*width)�������Ƕ�˼����
% ÿ���㶼��һ�����ʣ�����1/rate���Ÿ��£���ͬ������ͼͼ��ÿ�λ����
% һ�����ʲ��ָ��㣬��1/rate*height*width������£���������ļ��㸴
% �ӶȾͻή��O(1/rate*height*width)��
size =2*max(height, width) + 1; 
% jump[] from 1 to 2*updateFactor
% ������ʵ�ֵ�ʱ����ͨ����������ķ�ʽ��
% ���磬����Ϊ1~(2*rate - 1)֮���һ���������
% ÿ�θ���һ�������ǰ�ƶ�һ������������������
% ͼ��洢Ϊһ��һά���飬ÿ��ͨ���Ӳ���������
% ������������������ƽ��ÿ���ƶ��Ĳ���Լ����rate��
% ��ô����ͼ��ͻ���1/rate*height*width���и��£��ȼ�ʵ��
jump = floor(rand(1, size)*2*updateFactor) + 1;%%�������
% neighborX, Y represent the neighbor index
neighborX = floor(rand(1, size)*3) - 1;% -1~1
neighborY = floor(rand(1, size)*3) - 1;
% position[] from 1 to numberOfSamples
position = floor(rand(1, size)*numberOfSamples) + 1;  %1~20

disp('Initialize ViBe')