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

%% y 处理成one-hot向量(列向量), 用于forward propagation:

yOneHot = zeros(num_labels, size(y, 1));    % 参数num_labels: 有多少个分类.
% 注意 y 里面第0类是标记为10, 所以yOneHot里面[0, ... 0, 1]'表示第0类.
for i = 1 : size(y, 1)
    yOneHot(y(i, 1), i) = 1;
end


%% regularized forward propagation:
% 加bias item的1那一列.
X = [ones(size(X, 1), 1), X];
% 计算z2, 第二层.
z2 = X * Theta1';
% 计算a2.
a2 = sigmoid(z2);
% 加bias item.
a2 = [ones(size(a2, 1), 1), a2];
a3 = sigmoid(a2 * Theta2');

% 去掉bias item之后全部.^2即可.
tmpTheta1Sum = Theta1(:, 2 : end) .^2;
tmpTheta2Sum = Theta2(:, 2 : end) .^2;

% 计算 -cost, 注意下面只能用.*, 因为是每个用例预测和对应的y真实向量每个相乘.
costNeg = log(a3) .* (-yOneHot') - log(1 - a3) .* (1 - yOneHot');
J = sum(costNeg(:)) / m ...
    + (sum(tmpTheta1Sum(:)) + sum(tmpTheta2Sum(:))) * lambda / 2 / m;


for i = 1 : m
    % First, we do forward propogation on an X that already contains
    % the bias node (from above)

    a1 = X(i, :);
    z2 = Theta1 * a1';
        
    a2 = sigmoid(z2);
    a2 = [1; a2];
    
    % Now we have our final activation layer a3 == h(theta)
    a3 = sigmoid(Theta2 * a2);
    
    % Now that we have our activation layer, we go backwards
    % This basically just involves following along the formulas given
    % on Page 9
    d3 = a3 - yOneHot(:, i);
    
    % Re-add a bais node for z2
    z2 = [1; z2];
    d2 = (Theta2' * d3) .* sigmoidGradient(z2);
    % Strip out bais node from resulting d2
    d2 = d2(2 : end);

    Theta2_grad = Theta2_grad + d3 * a2';
    Theta1_grad = Theta1_grad + d2 * a1;
end

% Now divide everything (element-wise) by m to return the partial
% derivatives. Note that for regularization these will have to
% removed/commented out.

% Theta2_grad = Theta2_grad ./ m;
% Theta1_grad = Theta1_grad ./ m;

% -------------------------------------------------------------

% Implement Part III -- Regularization with cost function/gradients
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% The formula for regularization is given on page 12 and is as
% follows: Delta(l(i,j)) = 1/m*delta(l(i,j)) + lambda/m*(Theta(l(i,j))
% for j >= 1

% Implement for Theta1 and Theta2 when l = 0
Theta1_grad(:, 1) = Theta1_grad(:, 1) ./ m;
Theta2_grad(:, 1) = Theta2_grad(:, 1) ./ m;

% Implement for Theta1 and Theta2 when l > 0
Theta1_grad(:, 2 : end) = Theta1_grad(:, 2 : end) ./ m ...
    + ((lambda / m) * Theta1(:, 2 : end));
Theta2_grad(:, 2 : end) = Theta2_grad(:, 2 : end) ./ m ...
    + ((lambda / m) * Theta2(:, 2 : end));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
