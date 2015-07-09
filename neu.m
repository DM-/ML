% This will be for neural nets
% Need the math lib as usual
source funcs.m;

% Now what I want to implement

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This section is for feedforward results finding
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% A single neuron, takes data (row for sets, columns for variables) and weights (column) and activation function. func in @func form.
% Returns column of results
function results = snr(data,weights,func);
	results = func(data*weights);
end
% A layer of neurons, takes the same data as above, weights (each neurons weights take a column), derives size of net from weights 
% and activation function. Same as above due to vectorization & matrix math. Results is matrix data rows by weights columns
function results = lnr(data,weights,func);
	results = func(data*weights);
end
% A standard 3 layer network, takes data as above, weights arranged as above + 3rd dimension for each layer, 
% derives size of net from weights and activation function. Weights is a cell array column vector of maxtrixes.
function results = trilnr(data,weights,func);
	resultsIntermediate = func(data*weights{1});
	results = func(resultsIntermediate*weights{2});

end
% An arbritary size network, data as above, weights as above, derives size as above, and activation function
function results = nlnr(data,weights,func);
	results = data;
	for ex = 1:size(weights,1)
		results = func(results*weights{ex});
	end
end
%Adding a byte of code to sanity check the weights. Each layer should have as many columns as the next has rows.

function wsc(weights)
	if iscell(weights)==0 % not a cell = wrong place
		error("That wasnt a cell")
	end
	for ex = 1:size(weights,1)-1 % check all except last since last outputs to exit not another layer
		if size(weights{ex},2) != size(weights{ex+1},1) %columns to rows
			error("Layer size mismatch")
		endgit 
	end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Single Neuron Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% source flinear.m % for learning for the @self func
% function deltaweights = sndw(dedo,dodn,dndw) would be the real one, but we'll make one that assumes the error function is
% total squared error, and that delta net input / delta weights = value associated with said weight
function deltaweights = sndw(targets,       % calculates results from data & weights since we need those anyways
							 data,weights,
							 func,invfunc),
	deltaweights = data'*((snr(data,weights,func)-targets)*invfunc(data*weights)); % data'*((results-targets)*dodn(net_input))
	% the part after the * gives a matrix such that M(1,1) is the error in the first neuron for the first set of variables
	% M(2,1) is the error in the first neuron for the second set
	% M(1,2) is the error for the second neuron for the first set
	% What the above does is it takes M(:,1) , a column vector of the errors in the first neuron for each set
	% And multiplies it with the data(:,1)' , the set of all the first variables from all sets
	% This gives us how much the weight should be adjuted as learned from each set, and because of how matrix mult works
	% It sums up all those adjustments together. The result can be directly subtracted from the input weights matrix
	% to give an updated result.

end
% source logo.m    % for learning with the @sigm func
% For others, simply derivate (delta-error/delta-weights)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%BackProp
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Going right to the big one, backprop for nlnr, arbritary size network.
