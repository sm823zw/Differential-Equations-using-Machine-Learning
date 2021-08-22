clc;
clear;
close all;

data = load('../Data/cylinder_nektar_wake.mat');
X_star = data.X_star;
t_star = data.t;
U_star = data.U_star;
p_star = data.p_star;

N = length(X_star);
T = length(t_star);

XX = repmat(X_star(:,1),1,T);
YY = repmat(X_star(:,2),1,T);
TT = transpose(repmat(t_star,1,N));

UU = reshape(U_star(:,1,:),N,T);
VV = reshape(U_star(:,2,:),N,T);
PP = p_star;

x = XX(:);
y = YY(:);
t = TT(:);
u = UU(:);
v = VV(:);
p = PP(:);

N_train = 5000;
idx = randperm(N*T, N_train);
x_train = x(idx,:);
y_train = y(idx,:);
t_train = t(idx,:);
u_train = u(idx,:);
v_train = v(idx,:);

dataset = [x_train,y_train,t_train,u_train,v_train];
