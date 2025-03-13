function g = sigmoidGradient(z)
%SIGMOIDGRADIENT 返回在 z 处计算的 sigmoid 函数的梯度
%   g = SIGMOIDGRADIENT(z) 计算在 z 处的 sigmoid 函数的梯度。 
%   无论 z 是矩阵还是向量，这都应该有效。
%   特别是，如果 z 是向量或矩阵，你应该返回每个元素的梯度。
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).
for i=1:size(z,2)
    g(i)=sigmoid(z(i)).*(1- sigmoid(z(i)));
end
% =============================================================
end
