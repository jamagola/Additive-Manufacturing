% Adaptive MPC work

plant_mdl = 'mpc_cstr_plant';
op = operspec(plant_mdl);

op.Inputs(1).u = 10;
op.Inputs(1).Known = true;

op.Inputs(2).u = 298.15;
op.Inputs(2).Known = true;

op.Inputs(3).u = 298.15;
op.Inputs(3).Known = true;

[op_point, op_report] = findop(plant_mdl,op);

x0 = [op_report.States(1).x;op_report.States(2).x];
y0 = [op_report.Outputs(1).y;op_report.Outputs(2).y];
u0 = [op_report.Inputs(1).u;op_report.Inputs(2).u;op_report.Inputs(3).u];

sys = linearize(plant_mdl, op_point);
sys = sys(:,2:3); % MIMO

Ts = 0.5;
plant = c2d(sys,Ts); %Discrete plant

plant.InputGroup.MeasuredDisturbances = 1;
plant.InputGroup.ManipulatedVariables = 2;
plant.OutputGroup.Measured = 1;
plant.OutputGroup.Unmeasured = 2; 
plant.InputName = {'Ti','Tc'};
plant.OutputName = {'T','CA'};

mpcobj = mpc(plant);
mpcobj.Model.Nominal = struct('X', x0, 'U', u0(2:3), 'Y', y0, 'DX', [0 0]);

Uscale = [30 50];
Yscale = [50 10];
mpcobj.DV(1).ScaleFactor = Uscale(1);
mpcobj.MV(1).ScaleFactor = Uscale(2);
mpcobj.OV(1).ScaleFactor = Yscale(1);
mpcobj.OV(2).ScaleFactor = Yscale(2);

mpcobj.Weights.OV = [0 1];

mpcobj.MV.RateMin = -2;
mpcobj.MV.RateMax = 2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

mdl = 'ampc_cstr_linearization';
open_system(mdl)

