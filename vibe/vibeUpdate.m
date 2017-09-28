function [historyImages, historyBuffer] = vibeUpdate(buffer, shadowMap,updatingMask, historyImages, historyBuffer, param, ...
    jump, neighborX, neighborY, position)
    %% Parameters
    height  = param.height;
    width   = param.width;
    numberOfHistoryImages   = param.numberOfHistoryImages;   
%     numberOfSamples=param.numberOfSamples;
%     n=param.n;
%     step=param.step;
%     i=param.i;
%     t=param.t;
%     %% Update Model
%     for indY = 2:height - 1
%         shift = floor(rand()*width) + 1;
%         indX = jump(shift) + 1;
%         while indX < width
%              value = buffer(indY, indX);
%             if updatingMask(indY, indX) == 0  %即为背景（黑色为0 白色为255）
%                
%             else %即为前景
%                  if(flags(indY,indX)==0)
%                      counters(indY,indX)=counters(indY,indX)+1;
%                      if(counters(indY,indX)>=ghosth)%则该点判定为持续
%                      sigmaB=var(historyBuffer(indY,indX));
%                      sigmaF=var
%                      end
%                  else
%                       value=originalImage(indY,indX);
%                     flags=flags*0;%%对新来一帧每隔像素点
%                     %%进行分析，如果改像素点为背景则直接使
%                     %%用新来一帧像素点进行更新，若该像素点
%                     %%为前景，且标识为1，则用保存的原始背景
%                     % 模型替换当前背景模型，且标识flag清零，存储器清空
%                     
%                  end
%             end
%                 if position(shift) <= numberOfHistoryImages
%                     
%                     historyImages{position(shift)}(indY, indX) = value;
%                     historyImages{position(shift)}...
%                         (indY + neighborY(shift), indX + neighborX(shift)) = value;
%                 else
%                     pos = position(shift) - numberOfHistoryImages;
%                     historyBuffer{pos}(indY, indX) = value;
%                     historyBuffer{pos}...
%                         (indY + neighborY(shift), indX + neighborX(shift)) = value;
%                 end
%            
%             shift = shift + 1;
%             indX = indX + jump(shift);
%         end
%     end

 %% Update Model
    for indY = 2:height - 1
        shift = floor(rand()*width) + 1;
        indX = jump(shift) + 1;
        while indX < width
            if updatingMask(indY, indX) == 0 &&shadowMap(indY,indX)==1%即为背景（黑色为0 白色为255）且不是阴影
                value = buffer(indY, indX);
                if position(shift) <= numberOfHistoryImages
                    historyImages{position(shift)}(indY, indX) = value;
                    historyImages{position(shift)}...
                        (indY + neighborY(shift), indX + neighborX(shift)) = value;
                else
                    pos = position(shift) - numberOfHistoryImages;
                    historyBuffer{pos}(indY, indX) = value;
                    historyBuffer{pos}...
                        (indY + neighborY(shift), indX + neighborX(shift)) = value;
                end
            end
            shift = shift + 1;
            indX = indX + jump(shift);
        end
    end
end
% N=5;                                             %考虑6帧的帧间差分法（需要读取前7帧）
% start=5;                                        %start=100，从第100+1帧开始连续读7帧
% threshold=50;
% for k=1+start:N+1+start                         %处理从第101到第107帧                   
%         avi(k).cdata=(avi(k).cdata);      %将彩色图像转换为灰度图像
%         %avi(k-start).cdata=avi(k).cdata;
% end
%   [hang,lie]=size(avi(1+start).cdata);            %以avi（1+start）.cdata的格式生成一个矩阵
%   
%   alldiff=zeros(hang,lie,N);                       %生成一个三维的矩阵alldiff用于存储最终的各个帧的差分结果        
%   
% for k=1+start:N+start
%   diff=abs(avi(k).cdata-avi(k+1).cdata);           %邻帧差分
% 
%            %idiff=diff>20;                          %二值化，阈值选择为20，阈值调整
%            idiff=diff>threshold;                           %idiff中的数据位逻辑值，diff中的数值为unit8
%            alldiff(:,:,k)=double(idiff);           %存储各帧的差分结果，这里为什么要转换成double型的？？？？？
% end
% end