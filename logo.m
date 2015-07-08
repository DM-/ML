% Learn from experience E how to perform better at task T as measured by C
% we rely on V for @sigm etc
source funcs.m 
% Calculate results given weights+variables
function results = LogR(weights,variables);
	variables = [ones(length(variables),1),variables];
	if size(variables)(2) != size(weights)(1)
		error("wrong weights to variables size")
	end
	results  = sigm(variables*weights); 
end
% Cost function
function Cost = LogC(results,targets),
	m=size(results,1);
	Cost=(1/m)*sum(-targets.*log(results)-(1-targets).*log(1-results));
end;
% Gradient Descent 1-Batch Step
function DeltaWeights = LogDW(weights,variables,targets);
	% checking this before we add the bias 'variable'
	if size(targets)(1)!=size(variables)(1);
		error("Input size mismatch");
	end
	% using this since we wont ever add bias 'variable' to targets
	m = length(targets);

	% calculating results all in 1 go
	results = LogR(weights,variables);
	% Doing this here so I can unify how I handle bias and weights.
	variables = [ones(length(variables),1),variables];
	DeltaWeights = (variables'*(results-targets));
end
% Gradient Descent
function FinalWeights = LogGD(weights,variables,targets,lRate,nIters);
	iters = 0;
	%tempweights
	tWeights = weights;
	while iters < nIters;
		tWeights -= lRate*LogDW(tWeights,variables,targets);
		iters+=1;
	end
	FinalWeights = tWeights;
end
% Example Data
x1 = [1,2,3,2,1,2,2,3,1];
x2 = [5,6,4,4,5,4,6,4,5];
y  = [1,0,1,1,0,0,1,1,1]';
XXX=[x1',x2'];