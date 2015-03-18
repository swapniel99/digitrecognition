function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size1, ...
                                   hidden_layer_size2, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a three layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
l1 = 1;
r1 = hidden_layer_size1 * (input_layer_size + 1);
Theta1 = reshape(nn_params(l1:r1), ...
                 hidden_layer_size1, (input_layer_size + 1));

l2 = r1 + 1;
r2 = r1 + hidden_layer_size2 * (hidden_layer_size1 + 1);
Theta2 = reshape(nn_params(l2:r2), ...
                 hidden_layer_size2, (hidden_layer_size1 + 1));

l3 = r2 + 1;
Theta3 = reshape(nn_params(l3:end), ...
                 num_labels, (hidden_layer_size2 + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));

Y = zeros(m,num_labels);
for i = 1:m
    Y(i,y(i))=1;
end

A1 = [ones(m,1) X];
Z2 = A1*Theta1';
A2 = [ones(m,1) sigmoid(Z2)];
Z3 = A2*Theta2';
A3 = [ones(m,1) sigmoid(Z3)];
A4 = sigmoid(A3*Theta3');
H = A4;

J = sum(sum(-Y.*log(H)-(1-Y).*log(1-H)))/m ...
    + lambda*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) + sum(sum(Theta3(:,2:end).^2)))/(2*m);

D4 = A4-Y;	% m x 10
D3 = (D4*Theta3).*sigmoidGradient([ones(m,1) Z3]); % m x 26
D3 = D3(:,2:end); % m x 25
D2 = (D3*Theta2).*sigmoidGradient([ones(m,1) Z2]); % m x 26
D2 = D2(:,2:end); % m x 25

Theta3_grad = (D4'*A3)/m;
Theta2_grad = (D3'*A2)/m;
Theta1_grad = (D2'*A1)/m;

Theta3_grad(:,2:end) = Theta3_grad(:,2:end) + lambda*Theta3(:,2:end)/m;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda*Theta2(:,2:end)/m;
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda*Theta1(:,2:end)/m;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];

end
