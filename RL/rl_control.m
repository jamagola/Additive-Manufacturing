%%
% Initialization
clc; clear all; close all;
% Parameters
% a=0.1;
% b=0.25;
% A=1;
% LP=0; %0~5
% MD=0; %0~3
% r=0.8;
num=[0.2466 0.05715 0.001018];
den=[1 0.1855 0.003452];
% Plant internal delay must be smaller than simulink block sample rates
% NN model must have way to set initial conditions for this to work!
% **********************************************************************
obsInfo = rlNumericSpec([3 1],'LowerLimit',[-inf -inf 0  ]','UpperLimit',[ inf  inf inf]'); %%% inf to 2
obsInfo.Name = 'observations';
obsInfo.Description = 'integrated error, error, and measured response';
numObservations = obsInfo.Dimension(1);

actInfo = rlNumericSpec([1 1], 'LowerLimit',[0]','UpperLimit',[inf]'); %%% inf to 5
actInfo.Name = 'action';
numActions = actInfo.Dimension(1);

env = rlSimulinkEnv('rl_ddgp','rl_ddgp/RL Agent',obsInfo,actInfo);
env.ResetFcn = @(in)localResetFcn(in);

Ts = 0.1;
Tf = 100;
rng(0);

%%
% Critic
statePath = [
    featureInputLayer(numObservations,'Normalization','none','Name','State') 
    fullyConnectedLayer(50,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(25,'Name','CriticStateFC2')];
actionPath = [
    featureInputLayer(numActions,'Normalization','none','Name','Action')
    fullyConnectedLayer(25,'Name','CriticActionFC1')];
commonPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','CriticOutput')];

criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');

figure
plot(criticNetwork)

criticOpts = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1);
critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},criticOpts);
%%
% Actor
actorNetwork = [
    featureInputLayer(numObservations,'Normalization','none','Name','State')
    fullyConnectedLayer(3, 'Name','actorFC')
    tanhLayer('Name','actorTanh')
    fullyConnectedLayer(numActions,'Name','Action')
    ];

actorOptions = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1);

actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},actorOptions);

%%
% DDPG agent
agentOpts = rlDDPGAgentOptions(...
    'SampleTime',Ts,...
    'TargetSmoothFactor',1e-3,...
    'DiscountFactor',1.0, ...
    'MiniBatchSize',64, ...
    'ExperienceBufferLength',1e6); 
%agentOpts.NoiseOptions.StandardDeviation = 0.3;
%agentOpts.NoiseOptions.StandardDeviationDecayRate = 1e-5;

agent = rlDDPGAgent(actor,critic,agentOpts);

%%
% Train
maxepisodes = 100;
maxsteps = 1000;
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes, ...
    'MaxStepsPerEpisode',maxsteps, ...
    'ScoreAveragingWindowLength',20, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',20000);

doTraining = true; %%%%%%%%%%%%%%%%%%%%%%IMPORTANT%%%%%%%%%%%%%%%%%%%%%%%

if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
    % save('ddpgAgent.mat",'agent')
else
    % Load the pretrained agent for the example.
    load('ddpgAgent.mat','agent')
end

simOpts = rlSimulationOptions('MaxSteps',maxsteps,'StopOnError','on');
experiences = sim(env,agent,simOpts);

%%
function in = localResetFcn(in)

% Randomize reference signal
blk = sprintf('rl_ddgp/reference0');
r = abs(randn);
while r <= 0 || r >= 1
    r = abs(randn);
end
%r
in = setBlockParameter(in,blk,'After',num2str(r));

% Randomize initial height
% h = 3*randn + 10;
% while h <= 0 || h >= 20
%     h = 3*randn + 10;
% end
% blk = 'rl_ddgp/System/response0';
% in = setBlockParameter(in,blk,'InitialCondition',num2str(h));

end