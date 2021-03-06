clc;
clear;
close all;

% Load Data.
data = load('../Data/data_noisy.mat').data;
X = data(:, 1);
T = data(:, 2);
U = data(:, 3);
data = arrayDatastore(data);
%% 

% Construct the NN architecture and initialize the weights and learnable
% parameters.
n_layers = 9;
n_neurons = 20;

params = struct;

sz = [n_neurons 2];
params.fc1.Weights = initializeHe(sz,2);
params.fc1.Bias = initializeZeros([n_neurons 1]);

for layer_no=2:n_layers-1
    name = "fc"+layer_no;

    sz = [n_neurons n_neurons];
    n_inp = n_neurons;
    params.(name).Weights = initializeHe(sz,n_inp);
    params.(name).Bias = initializeZeros([n_neurons 1]);
end

sz = [1 n_neurons];
n_inp = n_neurons;
params.("fc" + n_layers).Weights = initializeHe(sz,n_inp);
params.("fc" + n_layers).Bias = initializeZeros([1 1]);

params.lambda1 = dlarray(0);
params.lambda2 = dlarray(0);
%% 

% Training hyper-paramaters
n_epochs = 30000;
mini_batch_size = 500;

initial_learning_rate = 0.01;
decay_rate = 0.001;
execution_environment = "auto";

% Create mini-batch iterator
mbq = minibatchqueue(data, ...
    'MiniBatchSize',mini_batch_size, ...
    'MiniBatchFormat','BC', ...
    'OutputEnvironment',execution_environment);

dlX = dlarray(X,'CB');
dlT = dlarray(T,'CB');
dlU = dlarray(U,'CB');

% Initialize arrays to store info about gradients required during adam
% optimization
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

% Start training
for epoch = 1:n_epochs
    % Reset and shuffle training data
    reset(mbq);
    while hasdata(mbq)
        iteration = iteration + 1;
        
        % Fetch new batch for training
        dlXT = next(mbq);
        dlX = dlXT(1,:);
        dlT = dlXT(2,:);
        dlU = dlXT(3,:);

        % Compute the gradients and loss
        [gradients,loss] = dlfeval(accfun,params,dlX,dlT,dlU);

        % Perform learning rate decay
        learning_rate = initial_learning_rate/ (1+decay_rate*iteration);

        % Perform Adam optimization and update the parameters
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

% Test the model.
t_test = [0.25 0.5 0.75 1];
n_predictions = 1001;
X_test = linspace(-1,1,n_predictions);

figure

for i=1:numel(t_test)
    t = t_test(i);
    TTest = t*ones(1,n_predictions);

    dlX_test = dlarray(X_test,'CB');
    dlT_test = dlarray(TTest,'CB');
    dlU_pred = model(params,dlX_test,dlT_test);

    U_test = solve_burgers(X_test,t,0.01/pi);

    error = norm(extractdata(dlU_pred) - U_test) / norm(U_test);

    subplot(2,2,i)
    plot(X_test,extractdata(dlU_pred),'-','LineWidth',2);
    ylim([-1.1, 1.1])

    hold on
    plot(X_test, U_test, '--','LineWidth',2)
    hold off

    title("t = " + t + ", Error = " + gather(error));
end

subplot(2,2,2)
legend('Predicted','True')
%% 

% Function that returns the true solution of Burger's equation at times t.
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

% Function that returns the gradients and loss for given batch of training samples. 
function [gradients,loss] = modelGradients(params,dlX,dlT,dlU)
    
    % Obtain U by forward-prop.
    U = model(params, dlX, dlT);
    
    % Compute loss_U.
    loss_U = mse(U, dlU);

    % Obtain gradients of U w.r.t X and T and Ux w.r.t X.
    gradientsU = dlgradient(sum(U,'all'),{dlX,dlT},'EnableHigherDerivatives',true);
    Ux = gradientsU{1};
    Ut = gradientsU{2};
    Uxx = dlgradient(sum(Ux,'all'),dlX,'EnableHigherDerivatives',true);
    
    % Compute U.
    f = Ut + params.lambda1.*dlarray(U.*Ux) - params.lambda2.*dlarray(Uxx);
    zero_target = zeros(size(f), 'like', f);
    
    % Compute loss_F.
    loss_F = mse(f, zero_target);

    % Compute total loss.
    loss = loss_F + loss_U;

    % Obtain gradients of loss w.r.t each learnable parameter.
    gradients = dlgradient(loss, params);
end

% Function that performs forward-prop.
function dlU = model(params,dlX,dlT)
    dlXT = [dlX;dlT];
    n_layers = numel(fieldnames(params))-2;
    weights = params.fc1.Weights;
    bias = params.fc1.Bias;
    dlU = fullyconnect(dlXT,weights,bias);
    for i=2:n_layers
        name = "fc" + i;
        dlU = tanh(dlU);
        weights = params.(name).Weights;
        bias = params.(name).Bias;
        dlU = fullyconnect(dlU, weights, bias);
    end
end

% Function to perform 'He' initialization of weights.
function weights = initializeHe(sz,n_inp)
    weights = randn(sz,'single') * sqrt(2/n_inp);
    weights = dlarray(weights);
end

% Function to perform 'Zero' initialization of bias or weights.
function parameter = initializeZeros(sz)
    parameter = zeros(sz,'single');
    parameter = dlarray(parameter);
end