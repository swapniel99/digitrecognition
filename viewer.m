clear; close all; clc;
test = csvread('test.csv');
res = csvread('res.csv');

m = size(res,1);

colormap(gray);

for i = 1:m
	clc;
	imagesc(reshape(test(i,:),28,28)');
	fprintf('Label predicted for image: %d',res(i,2));
	pause;
end
