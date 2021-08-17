clc;
clear all;
close all;

nu = 0.01/pi;
vxn = 50;
vtn = 50;
n = 2500;
vx = [-1 + (2) .* rand(1,vxn/2), linspace(-1,1,vxn/2)];
% vx = [-1 + (2) .* rand(1,vxn)];
vt = [rand(1,vtn/2), linspace(0.0001,1,vtn/2)];
% vt = [rand(1,vtn)];
data = zeros(n, 3);

i = 1;
for yi = 1:vtn
    u = solve_burgers(vx,vt(yi),nu);
    for xi = 1:vxn
        data(i, 1) = vx(xi);
        data(i, 2) = vt(yi);
        data(i, 3) = u(xi);
        i = i+1;
    end
end
ir = randperm(n);
data = data(ir,:);
scatter(data(:,1), data(:,2), 'filled', 'LineWidth', 1)
csvwrite('data.csv', data);


function U = solve_burgers(X,t,nu)
    f = @(y) exp(-cos(pi*y)/(2*pi*nu));
    g = @(y) exp(-(y.^2)/(4*nu*t));
    U = zeros(size(X));
    for i = 1:numel(X)
        x = X(i);
        if abs(x) ~= 1
            fun = @(eta) sin(pi*(x-eta)) .* f(x-eta) .* g(eta);
            uxt = -integral(fun,-inf,inf);
            fun = @(eta) f(x-eta) .* g(eta);
            U(i) = uxt / integral(fun,-inf,inf);
        end
    end
end
