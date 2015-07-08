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
function Cost = slogCFunc(results,targets),
	m=size(results,1);
	Cost=(-1/m)*sum(targets.*log(results)+(1-targets).*log(1-results));
end;
% Gradient Descent 1-Batch Step

% Gradient Descent

% Example Data