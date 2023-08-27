%%
% Initialization
clc; clear all; close all;
% Parameters
a=1;
b=1;
A=1;

obsInfo = rlNumericSpec([3 1],'LowerLimit',[-inf -inf 0  ]','UpperLimit',[ inf  inf inf]');
obsInfo.Name = 'observations';
obsInfo.Description = 'integrated error, error, and measured response';
numObservations = obsInfo.Dimension(1);

actInfo = rlNumericSpec([1 1]);
actInfo.Name = 'action';
numActions = actInfo.Dimension(1);

env = rlSimulinkEnv('rl_ddgp','rl_ddgp/RL Agent',obsInfo,actInfo);
env.ResetFcn = @(in)localResetFcn(in);

Ts = 1.0;
Tf = 200;
rng(0);

function in = localResetFcn(in)

% Randomize reference signal
blk = sprintf('rl_ddgp/reference0');
h = 3*randn + 10;
while h <= 0 || h >= 20
    h = 3*randn + 10;
end
in = setBlockParameter(in,blk,'Value',num2str(h));

% Randomize initial height
h = 3*randn + 10;
while h <= 0 || h >= 20
    h = 3*randn + 10;
end
blk = 'rl_ddgp/System/Response';
in = setBlockParameter(in,blk,'InitialCondition',num2str(h));

end