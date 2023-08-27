% LQR/LQG work

load lqrpilot
lqrdes

[K,~,~]=lqr(A_aug,B_aug,Q,R);

% A_aug created by adding integration of error as one state (1st).
% Hence, first row is all empty but last element -1.

% B_aug by adding top empty row

% Q, R diagonal matrix with entries to emphasize performance and effort
% respectively. 

