function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X1=X*Theta1(:,2:4).';
X1=X1+repmat(Theta1(:,1).',5,1);
G1=X1;
X1=1./(1+exp(-X1));
sigmoid_1=X1;
sigmoid_grad=X1.*(1-X1);
X2=X1*Theta2(:,2:6).';
X2=X2+repmat(Theta2(:,1).',5,1);
X2=1./(1+exp(-X2));
G2=X2;

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



%convert labels to vectors
Y1=zeros(5,3);
for i=1:5
    switch(y(i))
        case 1
            Y1(i,:)=[1 0 0];
        case 2
            Y1(i,:)=[0 1 0];
        case 3
            Y1(i,:)=[0 0 1];
    end
end

%unregularized cost function
if lambda==0
for i=1:5
    for k=1:3
    
    J=J-Y1(i,k)*log(X2(i,k))-(1-Y1(i,k))*log(1-X2(i,k));
    end
end
J=J/5;
%bias terms are not regularized

%add regularization term
else
for i=1:5
    for k=1:3
    
    J=J-Y1(i,k)*log(X2(i,k))-(1-Y1(i,k))*log(1-X2(i,k));
    end
end
J=J/5;    
    S=0;
    for i=1:5
        for j=1:3
            S=S+Theta1(i,j+1)*Theta1(i,j+1);
        end
    end
    for i=1:3
        for j=1:5
            S=S+Theta2(i,j+1)*Theta2(i,j+1);
        end
    end
    S=S*lambda/2/5;
    J=J+S;
end


%unregularized gradient

delta3=X2-Y1;
delta3=delta3.';
delta2=((Theta2(:,2:end).')*delta3).*(sigmoid_grad.');
%z=sum(z,1);
%z=z/5;
Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+(delta3)*(sigmoid_1);
Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+(delta2)*(X);
Theta2_grad=Theta2_grad/4000;
Theta1_grad=Theta1_grad/4000;
% -------------------------------------------------------------
%regularize gradient
if lambda~=0
    Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+lambda/5*Theta2(:,2:end);
    Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+lambda/5*Theta1(:,2:end);
% =========================================================================
end
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
