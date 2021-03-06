% Learn from experience E how to perform better at task T as measured by C
% any no of variables. 2 weights (As a column vector [Bias;weight1;weight2;etc]) + X variables (Also column vector). Output is column vector
source funcs.m  %math library, here to provide regress.
% Calculate results given weights+data.
function results = LinR(weights,variables);
	%variables = [ones(length(variables),1),variables];
	if size(variables)(2) != size(weights)(1)
		error("wrong weights to variables size")
	end
	results  = variables*weights; 
end
% Cost function, cost given results & targets.
% Mean Squared Error cost function.
% Input data in column vectors
function tc = LinC(results,targets);
	if length(targets)!=length(results);
		error("Input size mismatch");
	end
	m = length(results);
	tc = 0.5/m*sum((targets-results).^2);
end
% Gradient descent for the above.
% One batch weight change
% Actually more just returns derivative of above MSECSL
function DeltaWeights = LinDW(weights,variables,targets);
	% checking this before we add the bias 'variable'
	if size(targets)(1)!=size(variables)(1);
		error("Input size mismatch");
	end
	% using this since we wont ever add bias 'variable' to targets
	m = length(targets);

	% calculating results all in 1 go
	results = LinR(weights,variables);
	% Doing this here so I can unify how I handle bias and weights.
	% variables = [ones(length(variables),1),variables];
	DeltaWeights = 1/m.*(variables'*(results-targets));
end

% Multibatch

function FinalWeights = LinGD(weights,variables,targets,lRate,nIters);
	iters = 0;
	%tempweights
	tWeights = weights;
	while iters < nIters;
		reg(tWeights,lRate,0.001,size(targets,1)); %does regression , 0.001 is the regression rate, for later modification
		tWeights -= lRate*LinDW(tWeights,variables,targets);
		iters+=1;
	end
	FinalWeights = tWeights;
end

% Stop forgetting to add bias noob.
x1y1x=1:1:10;
x1y1y=1:1:10;
x1y1z=2:2:20;

x2y1x=1:1:10;
x2y1y=1:1:10;
x2y1z=3:3:30;

x2y0x=1:1:10;
x2y0y=1:1:10;
x2y0z=2:2:20;

%convient shortcut
%  [x2y0x',x2y0y']

X1=[2.9,2.4,2,2.3,3.2,1.9,3.4,2.1];
X2=[9.2,8.7,7.2,8.5,9.6,6.8,9.7,7.9];
X3=[13.2,11.5,10.8,12.3,12.6,10.6,14.1,11.2];
X4=[2,3,4,2,3,5,1,3]';
XXX=prepad([X1',X2',X3'],size([X1',X2',X3'],2)+1,1,2);

datalins = load('datalins.txt');
xlins = datalins(:, 1); ylins = datalins(:, 2);
xlins = zscore(xlins,1,1); % (X-mean)/std
xlins = prebias(xlins)    % add bias terms
llins = length(ylins);    % number of training examples

datalinm = load('datalinm.txt');
xlinm = datalinm(:, 1:2);
xlinm = zscore(xlinm,1,1); % (X-mean)/std
xlinm = prebias(xlinm);    % add bias terms
ylinm = datalinm(:, 3);
llinm = length(ylinm);
