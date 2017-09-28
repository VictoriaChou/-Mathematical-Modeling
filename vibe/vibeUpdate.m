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
%             if updatingMask(indY, indX) == 0  %��Ϊ��������ɫΪ0 ��ɫΪ255��
%                
%             else %��Ϊǰ��
%                  if(flags(indY,indX)==0)
%                      counters(indY,indX)=counters(indY,indX)+1;
%                      if(counters(indY,indX)>=ghosth)%��õ��ж�Ϊ����
%                      sigmaB=var(historyBuffer(indY,indX));
%                      sigmaF=var
%                      end
%                  else
%                       value=originalImage(indY,indX);
%                     flags=flags*0;%%������һ֡ÿ�����ص�
%                     %%���з�������������ص�Ϊ������ֱ��ʹ
%                     %%������һ֡���ص���и��£��������ص�
%                     %%Ϊǰ�����ұ�ʶΪ1�����ñ����ԭʼ����
%                     % ģ���滻��ǰ����ģ�ͣ��ұ�ʶflag���㣬�洢�����
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
            if updatingMask(indY, indX) == 0 &&shadowMap(indY,indX)==1%��Ϊ��������ɫΪ0 ��ɫΪ255���Ҳ�����Ӱ
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
% N=5;                                             %����6֡��֡���ַ�����Ҫ��ȡǰ7֡��
% start=5;                                        %start=100���ӵ�100+1֡��ʼ������7֡
% threshold=50;
% for k=1+start:N+1+start                         %����ӵ�101����107֡                   
%         avi(k).cdata=(avi(k).cdata);      %����ɫͼ��ת��Ϊ�Ҷ�ͼ��
%         %avi(k-start).cdata=avi(k).cdata;
% end
%   [hang,lie]=size(avi(1+start).cdata);            %��avi��1+start��.cdata�ĸ�ʽ����һ������
%   
%   alldiff=zeros(hang,lie,N);                       %����һ����ά�ľ���alldiff���ڴ洢���յĸ���֡�Ĳ�ֽ��        
%   
% for k=1+start:N+start
%   diff=abs(avi(k).cdata-avi(k+1).cdata);           %��֡���
% 
%            %idiff=diff>20;                          %��ֵ������ֵѡ��Ϊ20����ֵ����
%            idiff=diff>threshold;                           %idiff�е�����λ�߼�ֵ��diff�е���ֵΪunit8
%            alldiff(:,:,k)=double(idiff);           %�洢��֡�Ĳ�ֽ��������ΪʲôҪת����double�͵ģ���������
% end
% end