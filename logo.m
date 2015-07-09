% Learn from experience E how to perform better at task T as measured by C
% we rely on V for @sigm ,regress, etc
source funcs.m 
% Calculate results given weights+variables
function results = LogR(weights,variables);
	variables = prepad(variables,size(variables,1)+1,1,2);
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
	% calculating results all in 1 go
	results = LogR(weights,variables);
	% Doing this here so I can unify how I handle bias and weights.
	variables = prepad(variables,size(variables,1)+1,1,2);
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

data = load('data1.txt');
X2 = data(:, [1, 2]);
Y2 = data(:, 3);

load('data2.mat'); % training data stored in arrays X, y
sizeOfX = size(X, 1);
% Randomly select 100 data points to display
rand_indices = randperm(sizeOfX);
sel = X(rand_indices(1:100), :);

% Multiclass logistic regression
% Takes all variables LogGD does.
% Expects weights in the format [theta1classifier1,theta1classifier2,...;theta2classifier1,theta2classifier2,...] . ie like above
% but stuck next to each other.
% Variables should stay the same as above.
% From targets we will extract the different targets with unique(), and how many to make with length(), checking it with weights to catch errors
function FinalWeights = MCLogGD(weights,variables,targets,lRate,nIters);
	% Initialize variables by adding bias & creating container for results
	variables = prepad(variables,size(variables,1)+1,1,2);
	FinalWeights = zeros(size(weights));
	if size(targets)(1)!=size(variables)(1);
		error("Not equal no. of targets and variables");
	end
	if size(variables)(2) != size(weights)(1)
		error("wrong weights to variables size");
	end
	classes = unique(targets)
	if size(classes,1) != size(weights,2)
		error ("Expected no. of classes from weights vs classes in targets mismatch")
	end
	q=1
	for i = classes'
		localTargets = i == targets; % seperate it so everything is 0 except our class, at 1
		% need a way to maintain order of weights <-> class later for now use unique(targets) to find the order.
		localWeights = weights(:,q); % slice off out bit of the pie
		iters = 0;
		while iters < nIters;
			results = sigm(variables*localWeights); % get the results for this classifier
			localWeights -= lRate*(variables'*(results-localTargets));
			iters+=1;
		end
		FinalWeights(:,q)=localWeights;
		q+=1
	end
end