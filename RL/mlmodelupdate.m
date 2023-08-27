clear all; close all; clc;

csvfile='matlab_import.csv';
Table=readtable('matlab_import.csv');

data=table2array(Table);
train_percent=0.8;
test_index=round(length(data)*train_percent);
xtrain=data(1:test_index-1,1:3); % This may change
ytrain=data(1:test_index-1,6); % This may change
xtest=data(test_index:end,1:3); % This may change
ytest=data(test_index:end,6); % This may change

x=data(:,1:3); %Subject to change
y=data(:,6);

%nnstart: modelFFConfig

trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
hiddenLayerSize = [10,10];
net = fitnet(hiddenLayerSize,trainFcn);

net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
net.performFcn = 'mse';  % Mean Squared Error
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};

% Train the Network
[net,tr] = train(net,xtrain',ytrain');
% Test the Network
ypredict = net(xtest');
%ypredict=modelFF(xtest);

plot(ytest,'k');
hold on;
plot(ypredict','r');
legend('Actual', 'Predict');
grid on;
title('FeedForward')

gof=goodnessOfFit(ypredict',ytest,'NRMSE')
