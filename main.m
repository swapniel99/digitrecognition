%% Initialization
clear ; close all; clc;

%% Setup the parameters you will use for this exercise
hidden_layer_size1 = 25;   % 25 hidden units
hidden_layer_size2 = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

% Load Training Data
fprintf('Loading Data ...\n')

data = csvread('train.csv');
X = data(:,2:end);
y = data(:,1);
y(y==0) = 10;

m = size(X, 1);
input_layer_size = size(X, 2);

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size1);
initial_Theta2 = randInitializeWeights(hidden_layer_size1, hidden_layer_size2);
initial_Theta3 = randInitializeWeights(hidden_layer_size2, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:)];

%  You should also try different values of lambda
lambda = 1;

nn_params = trainNN(X,y,lambda,input_layer_size,hidden_layer_size1,hidden_layer_size2,num_labels,initial_nn_params);

% Obtain Theta1 and Theta2 back from nn_params
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

fprintf('Reading tests...\n')

testX = csvread('test.csv');
testm = size(testX,1);

pred = predict(Theta1, Theta2, Theta3, testX);
pred(pred==10) = 0;

fprintf('Writing results...\n')
res = [zeros(testm,1) pred];
for i = 1:testm
	res(i,1) = i;
end

csvwrite('res.csv',res);

%fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

