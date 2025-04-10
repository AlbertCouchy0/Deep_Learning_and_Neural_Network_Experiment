function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
% NNCOSTFUNCTION 实现了一个两层神经网络的成本函数，用于分类任务
%   [J grad] = NNCOSTFUNCTION(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
%   计算神经网络的成本和梯度。神经网络的参数被展开成向量nn_params，需要将其转换回权重矩阵。
%   返回的参数grad应是一个展开的向量，包含神经网络的各个偏导数。

% 将nn_params重新塑形成参数Theta1和Theta2，即两层神经网络的权重矩阵
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% 设置一些有用的变量
m = size(X, 1); % 样本数量
X1=X*Theta1(:,2:401).';
X1=X1+repmat(Theta1(:,1).',4000,1);
G1=X1;
X1=1./(1+exp(-X1));
sigmoid_1=X1;
sigmoid_grad=X1.*(1-X1);
X2=X1*Theta2(:,2:26).';
X2=X2+repmat(Theta2(:,1).',4000,1);
X2=1./(1+exp(-X2));
G2=X2;         

% 需要正确返回以下变量
J = 0; % 初始化成本为0
Theta1_grad = zeros(size(Theta1)); % 初始化Theta1的梯度
Theta2_grad = zeros(size(Theta2)); % 初始化Theta2的梯度

% ====================== YOUR CODE HERE ======================
%convert labels to vectors
Y1=zeros(4000,10);
for i=1:4000
    switch(y(i))
        case 1
            Y1(i,:)=[1 0 0 0 0 0 0 0 0 0];
        case 2
            Y1(i,:)=[0 1 0 0 0 0 0 0 0 0];
        case 3
            Y1(i,:)=[0 0 1 0 0 0 0 0 0 0];
        case 4
            Y1(i,:)=[0 0 0 1 0 0 0 0 0 0];
        case 5
            Y1(i,:)=[0 0 0 0 1 0 0 0 0 0];
        case 6
            Y1(i,:)=[0 0 0 0 0 1 0 0 0 0];
        case 7
            Y1(i,:)=[0 0 0 0 0 0 1 0 0 0];
        case 8
            Y1(i,:)=[0 0 0 0 0 0 0 1 0 0];
        case 9
            Y1(i,:)=[0 0 0 0 0 0 0 0 1 0];
        case 10
            Y1(i,:)=[0 0 0 0 0 0 0 0 0 1];
    end
end

%unregularized cost function
if lambda==0
for i=1:4000
    for k=1:10
    
    J=J-Y1(i,k)*log(X2(i,k))-(1-Y1(i,k))*log(1-X2(i,k));
    end
end
J=J/4000;
%bias terms are not regularized

%add regularization term
else
for i=1:4000
    for k=1:10
    
    J=J-Y1(i,k)*log(X2(i,k))-(1-Y1(i,k))*log(1-X2(i,k));
    end
end
J=J/4000;    
    S=0;
    for i=1:25
        for j=1:400
            S=S+Theta1(i,j+1)*Theta1(i,j+1);
        end
    end
    for i=1:10
        for j=1:25
            S=S+Theta2(i,j+1)*Theta2(i,j+1);
        end
    end
    S=S*lambda/2/4000;
    J=J+S;
end


%unregularized gradient

delta3=X2-Y1;
delta3=delta3.';
delta2=((Theta2(:,2:26).')*delta3).*(sigmoid_grad.');
%z=sum(z,1);
%z=z/4000;
Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+(delta3)*(sigmoid_1);
Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+(delta2)*(X);
Theta2_grad=Theta2_grad/4000;
Theta1_grad=Theta1_grad/4000;
% -------------------------------------------------------------
%regularize gradient
if lambda~=0
    Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+lambda/4000*Theta2(:,2:end);
    Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+lambda/4000*Theta1(:,2:end);
end
%========================================================================


% 展开梯度
grad = [Theta1_grad(:); Theta2_grad(:)];
end



%% 函数解读
%% 输入参数
% nn_params：神经网络的参数，展开成一个向量。
% input_layer_size：输入层的节点数。
% hidden_layer_size：隐藏层的节点数。
% num_labels：输出层的节点数，即类别数。
% X：输入数据，每一行是一个样本。
% y：样本的标签，每个标签是一个整数（从1到num_labels）。
% lambda：正则化参数，用于控制正则化的强度。

%% 输出参数
% J：神经网络的成本函数值。
% grad：神经网络的梯度，展开成一个向量。

%% 函数功能
% 该函数实现了两层神经网络的成本函数和梯度计算，具体步骤如下：

% 1.参数重塑：将输入的参数向量 nn_params 重塑成两个权重矩阵 Theta1 和 Theta2，分别对应输入层到隐藏层以及隐藏层到输出层的权重。

% 2.前向传播：
% 在输入数据 X 中添加偏置项，形成新的输入矩阵。
% 计算隐藏层的激活值 a2，使用 sigmoid 函数对输入数据与权重 Theta1 的乘积进行激活。
% 在隐藏层激活值中添加偏置项，形成新的矩阵。
% 计算输出层的激活值 a3，使用 sigmoid 函数对隐藏层激活值与权重 Theta2 的乘积进行激活。

% 3.标签转换：将标签 y 转换为 one-hot 编码形式，方便后续计算。

% 4.成本函数计算：
% 计算未正则化的成本函数，使用交叉熵损失函数。
% 处理偏置项，不进行正则化。
% 添加正则化项到成本函数，形成最终的成本值 J。

% 5.反向传播：
% 计算输出层的误差 d3。
% 计算隐藏层的误差 d2，使用 sigmoidGradient 函数（sigmoid 函数的导数）。
% 去掉偏置项的误差。
% 计算权重矩阵 Theta1 和 Theta2 的梯度。

% 6.梯度正则化：对梯度进行正则化处理，不包括偏置项。

% 7.梯度展开：将计算得到的梯度矩阵 Theta1_grad 和 Theta2_grad 展开成一个向量 grad，便于后续优化算法使用。

% 该函数在神经网络的训练过程中起着核心作用，通过计算成本和梯度，为优化算法（如梯度下降）提供必要的信息，以调整网络参数，最小化预测误差。