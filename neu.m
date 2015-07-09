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
% function deltaweights = sndw(dedo,dodn,dndw) would be the real one, but we'll make one that assumes the error function is
% total squared error, and that delta net input / delta weights = value associated with said weight
function deltaweights = lndw(targets,       % calculates results from data & weights since we need those anyways
							 data,weights,
							 func,invfunc),
	deltaweights = data'*((snr(data,weights,func)-targets).*invfunc(data*weights)); % data'*((results-targets)*dodn(net_input))
	% the part after the * gives a matrix  such that M(1,1) is the error in the first neuron for the first set of variables
	% M(2,1) is the error in the first neuron for the second set
	% M(1,2) is the error for the second neuron for the first set
	% What the above does is it takes M(:,1) , a column vector of the errors in the first neuron for each set
	% And multiplies it with the data(:,1)' , the set of all the first variables from all sets
	% This gives us how much the weight should be adjuted as learned from each set, and because of how matrix mult works
	% It sums up all those adjustments together. The result can be directly subtracted from the input weights matrix
	% to give an updated result.

end
function endweights = lngd(targets,       % calculates results from data & weights since we need those anyways
							 data,weights,
							 func,invfunc,
							 lRate,nIters),
	endweights = weights;
	for ex = 1:nIters
		endweights -= lRate*lndw(targets,data,endweights,func,invfunc);
	end
end
% source logo.m    % for learning with the @sigm func
% For others, simply derivate (delta-error/delta-weights)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%BackProp
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Going right to the big one, backprop for nlnr, arbritary size network.

% With lndw implemented we just need to adapt it, add a function that calculats error signal, and a function to combine them

% Function to do a forward pass and grab the net input going in at each layer & output from each layer.
% The output is in netoutcell(:,2), and net inputs in netoutcell(:,1)

function netoutcell = laynetout(data,weights,func) % thats all we need for a forward pass
	netoutcell = cell(size(weights,1)+1,2);
	netoutcell(1,1) =netoutcell(1,2) = data ; % Net = output for the input layer, since it has no activation function. Only output used
	for ex = 2:size(weights,1)+1 % skip first since thats for input layer
		netoutcell(ex,1) = netoutcell{ex-1,2}*weights{ex-1};  %The net input of layer x=output of layer x-1*weights connecting x-1 2 x 
		netoutcell(ex,2) = func(netoutcell{ex,1}); % The output of a layer is activation function (net input of layer)
	end
end
function deltaweights = lndw(targets,       % calculates results from data & weights since we need those anyways
							 data,weights,
							 func,invfunc),
	depth = size(weights,1); % How many weight matrixes are there? There are that many layers +1 input layer.
	if depth == 1
		error("For single layer networks please use lndw and lngd"); 	% Due to how the top layer has special handling, I decided this
																		% is easier
	end
	deltaweights = cell(size(weights,1),1); % output has as many matrixes as the weights input
	netoutcell   = laynetout(data,weights,func); % This shows us the gizzards of the net, does the feedforward and gets all we need.
	% deltaweights(x) is the weights from x to x+1, netoutcell(x) is the net input/output pair for layer x
	esig               = (netoutcell{depth+1,2}-targets).*invfunc(netoutcell{depth+1,1});
	deltaweights(depth)= netoutcell{depth,2}'*esig;
	% wl2l3 = l2.o * ((l3.o-l3.t).*f'(l3.n) where w is weights, l is layer, .o is output , .t is targets, .n is net input
	% and f' is derivative of the activation function. This is the formula for the output layer.
	for ex = depth-1:-1:1
		esig = (esig*weights{ex+1}').*invfunc(netoutcell{ex+1,1});
		deltaweights(ex)=netoutcell{ex,2}'*esig;
	end
end


