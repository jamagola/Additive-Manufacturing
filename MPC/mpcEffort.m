% Adaptive MPC effort
clc; clear all; close all;
% Define plant
% num=[0.2466 0.05715 0.001018];
% den=[1 0.1855 0.003452];
% Ts=0.1;
% H=tf(num,den);
% H_d=c2d(H,Ts);
% n=cell2mat(H_d.num);
% d=cell2mat(H_d.den);

%plant_mdl = 'amplant';
m=1;k=1;b=1.2;
plant_mdl = 'msd';
op = operspec(plant_mdl);

op.Inputs(1).u = 0;
op.Inputs(1).Known = true;

[op_point, op_report] = findop(plant_mdl,op);

x0 = [op_report.States(1).x]; 
y0 = [op_report.Outputs(1).y];
u0 = [op_report.Inputs(1).u];
% x0=0;
% y0=0;
% u0=0;

%sys=linearize(plant_mdl, op_point);
%sys.D=0;
sys=linearize(plant_mdl);
%sys=minreal(sys);
%ob=obsv(sys.A,sys.C);
%missing_state=length(sys.A)-rank(ob);
Ts = 0.1;
plant = c2d(sys,Ts); %Discrete plant
%plant = d2d(sys,Ts);
%plant=H_d;

%Indices
%plant.InputGroup.MeasuredDisturbances = 1;
plant.InputGroup.ManipulatedVariables = 1;
plant.OutputGroup.Measured = 1;
%plant.OutputGroup.Unmeasured = 1;
plant.InputName = {'LP'};
plant.OutputName = {'MP'};

mpcobj = mpc(plant);
mpcobj.Model.Nominal = struct('X', x0, 'U', u0, 'Y', y0, 'DX', [0]); %%%

Uscale = [1];
Yscale = [1];

mpcobj.MV(1).ScaleFactor = Uscale(1);
mpcobj.OV(1).ScaleFactor = Yscale(1);

mpcobj.Weights.OV = [1];

mpcobj.MV.RateMin = -100;
mpcobj.MV.RateMax = 100;

mpcobj.p=10;
mpcobj.Ts=0.1;
mpcobj.C=3;