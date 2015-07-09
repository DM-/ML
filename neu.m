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
		end
	end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Single Neuron Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% source flinear.m % for learning for the @self func
% source logo.m    % for learning with the @sigm func
% For others, simply derivate (delta-error/delta-weights)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%BackProp
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Going right to the big one, backprop for nlnr, arbritary size network.
