% Imported pretrained model
clear all; close all; clc;

csvfile='matlab_import.csv';
Table=readtable('matlab_import.csv');

data=table2array(Table);
train_percent=0.8;
test_index=round(length(data)*train_percent);
xtrain=data(1:test_index-1,1); % This may change
ytrain=data(1:test_index-1,6); % This may change
xtest=data(test_index:end,1); % This may change
ytest=data(test_index:end,6); % This may change

x=data(:,1); %Subject to change
y=data(:,6);

%nnstart: modelFFConfig

trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
hiddenLayerSize = 10;
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

%SID evaluation using transfer function

data_ = datastore(csvfile);
dt=10/1000; %10 ms

t=[0:dt:length(xtest)*dt-dt]';
u=xtest;
num=[0.2466 0.05715 0.001018];
den=[1 0.1855 0.003452];
H=tf(num,den)
ySID=lsim(H,u,t);

figure()
plot(ytest,'k');
hold on;
plot(ySID,'r');
legend('Actual', 'Predict');
grid on;
title('SID')

gof=goodnessOfFit(ySID,ytest,'NRMSE')

%nnstart: modelNARX
U=con2seq(xtrain'); % This may change
Y=con2seq(ytrain'); % This may change
Utest=con2seq(xtest'); % This may change
Ytest=con2seq(ytest'); % This may change

d1 = [1:2];
d2 = [1:2];
narx_net = narxnet(d1,d2,10);
narx_net.divideFcn = '';
narx_net.trainParam.min_grad = 1e-10;
[p,Pi,Ai,t] = preparets(narx_net,U,{},Y);

narx_net = train(narx_net,p,t,Pi);
narx_net_closed = closeloop(narx_net);

%[p1,Pi1,Ai1,t1] = preparets(narx_net,Utest,{},Ytest);
[p1,Pi1,Ai1,t1] = preparets(narx_net_closed,Utest,{},Ytest);
yNARX = narx_net_closed(p1,Pi1,Ai1);
figure()
plot(cell2mat(t1),'k');
hold on;
plot(cell2mat(yNARX),'r');
legend('Actual', 'Predict');
grid on;
title('NARX')
yNARX=cell2mat(yNARX);
t1=cell2mat(t1);
gof=goodnessOfFit(yNARX',t1','NRMSE')

%gensim(net, ts) %% Gate to simulink!!