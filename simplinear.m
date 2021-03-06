% Learn from experience E how to perform better at task T as measured by C
% Simple linear first. 2 weights (As a column vector [Bias;weight]) + 1 variable (Also column vector). Output is column vector
% Calculate results given weights+data.
function results = SLR(weights,variable);
	variables = prepad(variable,size(variable,2)+1,1,2);
	results  = variables*weights; 
end
% Draw line as plot, given weights data, use above to get results needed.
function PSLR(weights,variable);
	results = SLR(weights,variable);
	plot(variable,results);
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
function DeltaWeights = DWSLGD(weights,variable,targets);
	% checking this before we add the bias 'variable'
	if length(targets)!=length(variable);
		error("Input size mismatch");
	end
	% using this since we wont ever add bias 'variable' to targets
	m = length(targets);
	% container to fill with answer
	DeltaWeights=zeros(size(weights)); % so that they always match.
	% calculating results all in 1 go
	results = SLR(weights,variable);

	DeltaWeights(1)=1/m*sum(results-targets); % this is .*bias, but bias is always one in this case.
	DeltaWeights(2)=1/m*sum((results-targets).*variable);
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