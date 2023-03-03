# Hamiltonian Neural Network


Hamiltonian Neural Network[1] enables you to use Neural Networks under the law of conservation of energy.


<img src="https://insidelabs-git.mathworks.com/tfukumot/hamiltonian-neural-network/-/raw/main/Pics/1.png" width="720">


Hamiltonian Neural Network Loss is expressed with the following equation.
<img src="https://insidelabs-git.mathworks.com/tfukumot/hamiltonian-neural-network/-/raw/main/Pics/2.png" width="720">


## **Requirements**
- [MATLAB &reg;](https://jp.mathworks.com/products/matlab.html)
- [Deep Learning Toolbox<sup>TM</sup>](https://jp.mathworks.com/products/deep-learning.html)

MATLAB version should be R2022b and later (Tested in R2022b)

## **References**

  [1]  Sam Greydanus, Misko Dzamba, Jason Yosinski, Hamiltonian Neural Network, arXiv:1906.01563v1 [cs.NE] 4 Jun 2019. 1906.01563v1.pdf (arxiv.org) 

The data in 'trajectory_training.csv' was generated using Hamiltonian Neural Network described in the [paper](https://arxiv.org/abs/1906.01563) by Sam Greydanus, Misko Dzamba, Jason Yosinski , 2019, and released on [GitHub](https://github.com/greydanus/hamiltonian-nn) under an Apache 2.0 license.


# Demo_Hamiltonian_Spring_with_dlnetwork.m


## Import data

```matlab:Code
rng(0);
data = table2array(readtable("trajectory_training.csv"));
ds = arrayDatastore(dlarray(data',"BC"));
```


## Define Network

```matlab:Code
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
```


## Specify Training Options

```matlab:Code
numEpochs = 300;
miniBatchSize = 750;
executionEnvironment = "auto";
initialLearnRate = 0.001;
decayRate = 1e-4;
```


## Create a minibatchque

```matlab:Code
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
```


## Train model

```matlab:Code
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
```


## Test model



To make predictions with the Hamiltonian NN we need to solve the ODE system: dp/dt = -dH/dq, dq/dt = dH/dp



```matlab:Code
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
```


## Supporting Functions



modelGradients Function



```matlab:Code
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
```


*Copyright 2023 The MathWorks, Inc.*
