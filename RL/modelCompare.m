% Imported pretrained model
clear all; close all; clc;

modelfile='matlab_h5_model.h5';
net=importKerasNetwork(modelfile)
plot(net)

csvfile='matlab_import.csv'
Table=readtable('matlab_import.csv');

data=table2array(Table);
train_percent=0.8;
test_index=round(length(data)*train_percent);
x=data(test_index:end,2:6); % This may change
y=data(test_index:end,7); % This may change
y_=predict(net, x);
xtst=data(test_index:end,2); % This may change

plot(y,'k');
hold on;
plot(y_,'r');
legend('Actual', 'Predict');
grid on;


gof=goodnessOfFit(y_,y,'NRMSE')

figure()
y__=FFMATLAB(x);
plot(y,'k');
hold on;
plot(y__,'r');
legend('Actual', 'Predict');
grid on;


gof=goodnessOfFit(y__,y,'NRMSE')

% SID evaluation using transfer function

data_ = datastore(csvfile);
dt=10/1000; %10 ms

t=[0:dt:length(x)*dt-dt];
u=xtst;
num=[0.2466 0.05715 0.001018];
den=[1 0.1855 0.003452];
H=tf(num,den)
ySID=lsim(H,u,t);

figure()
plot(y,'k');
hold on;
plot(ySID,'r');
legend('Actual', 'Predict');
grid on;

gof=goodnessOfFit(ySID,y,'NRMSE')



