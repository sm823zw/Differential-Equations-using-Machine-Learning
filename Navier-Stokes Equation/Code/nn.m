clc;
clear;
close all;

data = load('../Data/data.mat').dataset;
X = data(:, 1);
Y = data(:, 2);
T = data(:, 3);
U = data(:, 4);
V = data(:, 5);
data = arrayDatastore(data);
%%

n_layers = 5;
n_neurons = 20;

params = struct;

sz = [n_neurons 3];
params.fc1.Weights = initializeHe(sz,3);
params.fc1.Bias = initializeZeros([n_neurons 1]);

for layer_no=2:n_layers-1
    name = "fc"+layer_no;

    sz = [n_neurons n_neurons];
    n_inp = n_neurons;
    params.(name).Weights = initializeHe(sz,n_inp);
    params.(name).Bias = initializeZeros([n_neurons 1]);
end

sz = [2 n_neurons];
n_inp = n_neurons;
params.("fc" + n_layers).Weights = initializeHe(sz,n_inp);
params.("fc" + n_layers).Bias = initializeZeros([2 1]);

params.lambda1 = dlarray(0);
params.lambda2 = dlarray(0);
%%

n_epochs = 20000;
mini_batch_size = 2500;

initial_learning_rate = 0.01;
decay_rate = 0.0005;
execution_environment = "gpu";

mbq = minibatchqueue(data, ...
    'MiniBatchSize',mini_batch_size, ...
    'MiniBatchFormat','BC', ...
    'OutputEnvironment',execution_environment);

dlX = dlarray(X,'CB');
dlY = dlarray(Y,'CB');
dlT = dlarray(T,'CB');
dlU = dlarray(U,'CB');
dlV = dlarray(V,'CB');

average_grad = [];
average_sq_grad = [];

accfun = dlaccelerate(@modelGradients);

figure
C = colororder;
line_loss = animatedline('Color',C(2,:));
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on

start = tic;
iteration = 0;

for epoch = 1:n_epochs
    reset(mbq);
    while hasdata(mbq)
        iteration = iteration + 1;
        dlXT = next(mbq);
        dlX = dlXT(1,:);
        dlY = dlXT(2,:);
        dlT = dlXT(3,:);
        dlU = dlXT(4,:);
        dlV = dlXT(5,:);

        [gradients,loss] = dlfeval(accfun,params,dlX,dlY,dlT,dlU,dlV);

        learning_rate = initial_learning_rate/ (1+decay_rate*iteration);

        [params,average_grad,average_sq_grad] = adamupdate(params,gradients,average_grad, ...
            average_sq_grad,iteration,learning_rate);
    end
    
    disp(epoch);
    disp(params.lambda1);
    disp(params.lambda2);
    loss = double(gather(extractdata(loss)));
    addpoints(line_loss,iteration, loss);

    D = duration(0,0,toc(start),'Format','hh:mm:ss');
    title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", Loss: " + loss)
    drawnow
end
%%
function [gradients,loss] = modelGradients(params,dlX,dlY,dlT,dlU,dlV)

    [psi, P] = model(params,dlX,dlY,dlT);
    U = dlgradient(sum(psi,'all'),dlY,'EnableHigherDerivatives',true);
    V = -dlgradient(sum(psi,'all'),dlX,'EnableHigherDerivatives',true);

    loss_U = mse(U, dlU);
    loss_V = mse(V, dlV);
    
    gradientsU = dlgradient(sum(U,'all'),{dlX,dlY},'EnableHigherDerivatives',true);
    Ux = gradientsU{1};
    Uy = gradientsU{2};
    Ut = dlgradient(sum(U,'all'), dlT);
    Uxx = dlgradient(sum(Ux,'all'),dlX,'EnableHigherDerivatives',true);
    Uyy = dlgradient(sum(Uy,'all'),dlY,'EnableHigherDerivatives',true);
    
    gradientsV = dlgradient(sum(V,'all'),{dlX,dlY},'EnableHigherDerivatives',true);
    Vx = gradientsV{1};
    Vy = gradientsV{2};
    Vt = dlgradient(sum(V,'all'), dlT);
    Vxx = dlgradient(sum(Vx,'all'),dlX,'EnableHigherDerivatives',true);
    Vyy = dlgradient(sum(Vy,'all'),dlY,'EnableHigherDerivatives',true);
    
    Px = dlgradient(sum(P,'all'),dlX);
    Py = dlgradient(sum(P,'all'),dlY);
    
    f = Ut + params.lambda1.*dlarray(U.*Ux + V.*Uy) + Px - params.lambda2.*dlarray(Uxx + Uyy);
    zero_target = zeros(size(f), 'like', f);
    loss_F = mse(f, zero_target);
    
    g = Vt + params.lambda1.*dlarray(U.*Vx + V.*Vy) + Py - params.lambda2.*dlarray(Vxx + Vyy);
    zero_target = zeros(size(g), 'like', g);
    loss_G = mse(g, zero_target);
    
    loss = loss_U + loss_V + loss_F + loss_G;

    gradients = dlgradient(loss, params);
end

function [dlpsi, dlP] = model(params,dlX,dlY,dlT)
    dlXT = [dlX;dlY;dlT];
    n_layers = numel(fieldnames(params))-2;
    weights = params.fc1.Weights;
    bias = params.fc1.Bias;
    out = fullyconnect(dlXT,weights,bias);
    for i=2:n_layers
        name = "fc" + i;
        out = tanh(out);
        weights = params.(name).Weights;
        bias = params.(name).Bias;
        out = fullyconnect(out, weights, bias);
    end
    dlpsi = out(1,:);
    dlP = out(2,:);
end

function weights = initializeHe(sz,n_inp)
    weights = randn(sz,'single') * sqrt(2/n_inp);
    weights = dlarray(weights);
end

function parameter = initializeZeros(sz)
    parameter = zeros(sz,'single');
    parameter = dlarray(parameter);
end