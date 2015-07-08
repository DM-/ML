% Learn from experience E how to perform better at task T as measured by C
% any no of variables. 2 weights (As a column vector [Bias;weight1;weight2;etc]) + X variable (Also column vector). Output is column vector

% Calculate results given weights+data.
function results = SLR(weights,variables);
	variables = [ones(length(variables),1),variables];
	if size(variables)(2) != size(weights)(1)
		error("wrong weights to variables size")
	end
	results  = variable*weights; 
end
% Draw line as plot, given weights data, use above to get results needed.
function PSLR(weights,variables);
	results = SLR(weights,variables);
	plot(variables,results);
end
% Cost function, cost given results & targets.
% Mean Squared Error cost function.
% Input data in column vectors
function tc = MSECSL(results,targets);
	if length(targets)!=length(results);
		error("Input size mismatch");
	end
	m = length(results);
	tc = 0.5/m*sum((targets-results).^2);
end
% Gradient descent for the above.
% One batch weight change
% Actually more just returns derivative of above MSECSL
function DeltaWeights = DWSLGD(weights,variables,targets);
	% checking this before we add the bias 'variable'
	if length(targets)!=length(variables);
		error("Input size mismatch");
	end
	% using this since we wont ever add bias 'variable' to targets
	m = length(targets);

	% calculating results all in 1 go
	results = SLR(weights,variables);
	% Doing this here so I can unify how I handle bias and weights.
	variables = [ones(length(variables),1),variables];

	DeltaWeights = 1/m*sum(variables'*results-targets));
end

% Multibatch

function FinalWeights = SLGDMB(weights,variable,targets,lRate,nIters);
	iters = 0;
	%tempweights
	tWeights = weights;
	while iters < nIters;
		tWeights -= lRate*DWSLGD(tWeights,variable,targets);
		iters+=1;
	end
	FinalWeights = tWeights;
end

%Examples
%slope 1 intercept 0
s1i0x=[1;2;3;4;5;6;7;8;9;10];
s1i0y=[1;2;3;4;5;6;7;8;9;10];
s1i0w=[0;1];
%slope 1 intercept 1
s1i1x=s1i0x;
s1i1y=s1i0y.+1;
s1i1w=[1;1];
%slope 2 intercept 0
s2i0x=s1i0x;
s2i0y=s1i0y*2;
s2i0w=[0;2];