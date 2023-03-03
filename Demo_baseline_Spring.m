%% Import data
rng(0);
data = table2array(readtable("trajectory_training.csv"));
ds = arrayDatastore(dlarray(data',"BC"));
%% Define Network

hiddenSize = 200;
inputSize = 2;
outputSize = 2;
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
%% Train model
tart = tic;

iteration = 0;
shuffle(mbq);

for epoch = 1:numEpochs
    reset(mbq);
%    shuffle(mbq);
    
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

    D = duration(0,0,toc(start),'Format','hh:mm:ss');
    title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", Loss: " + loss)
    drawnow
end
%% Test model

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

%% Supporting Functions
% modelGradients Function
function [gradients,loss] = modelGradients(net,dlX,dlT)
% Make predictions with the initial conditions.
dlT_pred = forward(net,dlX);

loss = mse(dlT_pred,dlT);
% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,net.Learnables);

end

% predmodel Function
function dlT_pred = predmodel(t,dlX,net)
    dlT_pred = forward(net,dlX);
end
%% 
% _Copyright 2023 The MathWorks, Inc._