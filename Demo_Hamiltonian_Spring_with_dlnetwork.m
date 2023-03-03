%% Import data
rng(0);
data = table2array(readtable("trajectory_training.csv"));
ds = arrayDatastore(dlarray(data',"BC"));
%% Define Network

hiddenSize = 200;
inputSize = 2;
outputSize = 1;
net = [
    featureInputLayer(inputSize)
    fullyConnectedLayer(hiddenSize)
    tanhLayer()
    fullyConnectedLayer(hiddenSize)
    tanhLayer()
    fullyConnectedLayer(outputSize)];
% Create a dlnetwork object from the layer array.
net = dlnetwork(net);
%% Specify Training Options

numEpochs = 300;
miniBatchSize = 750;
executionEnvironment = "auto";
initialLearnRate = 0.001;
decayRate = 1e-4;

%% Create a minibatchque
mbq = minibatchqueue(ds, ...
    'MiniBatchSize',miniBatchSize, ...
    'MiniBatchFormat','BC', ...
    'OutputEnvironment',executionEnvironment);
averageGrad = [];
averageSqGrad = [];

accfun = dlaccelerate(@modelGradients);

figure
C = colororder;
lineLoss = animatedline('Color',C(2,:));
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on
set(gca, 'YScale', 'log');
hold off
%% Train model
start = tic;

iteration = 0;
for epoch = 1:numEpochs
    shuffle(mbq);
    while hasdata(mbq)
        iteration = iteration + 1;

        dlXT = next(mbq);
        dlX = dlXT(1:2,:);
        dlT = dlXT(3:4,:);

        % Evaluate the model gradients and loss using dlfeval and the
        % modelGradients function.
        [gradients,loss] = dlfeval(accfun,net,dlX,dlT);
        % Update learning rate.
        learningRate = initialLearnRate / (1+decayRate*iteration);

        % Update the network parameters using the adamupdate function.
        [net,averageGrad,averageSqGrad] = adamupdate(net,gradients,averageGrad, ...
            averageSqGrad,iteration,learningRate);
    end

    % Plot training progress.
    loss = double(gather(extractdata(loss)));
    addpoints(lineLoss,iteration, loss);

    drawnow
end
%% Test model
% To make predictions with the Hamiltonian NN we need to solve the ODE system: 
% dp/dt = -dH/dq, dq/dt = dH/dp

accOde = dlaccelerate(@predmodel);
t0 = dlarray(0,"CB");
x = dlarray([1,0],"BC");
dlfeval(accOde,t0,x,net);

% Since the original ode45 can't use dlarray we need to write an ODE
% function that wraps accOde by converting the inputs to dlarray, and
% extracting them again after accOde is applied. 
f = @(t,x) extractdata(accOde(dlarray(t,"CB"),dlarray(x,"CB"),net));

% Now solve with ode45
x = single([1,0]);
t_span = linspace(0,20,2000);
noise_std =0.1;
% Make predictions.
t_span = t_span.*(1 + .9*noise_std);
[~,dlqp] = ode45(f,t_span,x); 
qp = squeeze(double(dlqp));
qp = qp.';
figure,plot(qp(1,:),qp(2,:))
hold on
load qp_baseline.mat
plot(qp(1,:),qp(2,:))
hold off
legend(["Hamiltonian NN","Baseline"])
xlim([-1.1 1.1])
ylim([-1.1 1.1])
%% Supporting Functions
% modelGradients Function
function [gradients,loss] = modelGradients(net,dlX,dlT)

% Make predictions with the initial conditions.
dlU = forward(net,dlX);
[dq,dp] = dlderivative(dlU,dlX);
loss_dq = l2loss(dq,dlT(1,:));
loss_dp = l2loss(dp,dlT(2,:));
loss = loss_dq + loss_dp;
gradients = dlgradient(loss,net.Learnables);
end

% predmodel Function
function dlT_pred = predmodel(t,dlX,net)
    dlU = forward(net,dlX);
    [dq,dp] = dlderivative(dlU,dlX);
    dlT_pred = [dq;dp];
end

% dlderivative Function
function [dq,dp] = dlderivative(F1,dlX)
dF1 = dlgradient(sum(F1,"all"),dlX);
dq = dF1(2,:);
dp = -dF1(1,:);
end
%% 
% _Copyright 2023 The MathWorks, Inc._