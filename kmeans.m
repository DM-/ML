% Time to implement kmeans
% Cluster assignment step, then move cluster 'centroid' to mean of it's assigned datapoints
% kmeans cost function, average of the sum of squared eucledian distance and its assigned centroid
% since both steps lower the cost, cost can only go down.
% This folder relies on broadcasting and so the warning is disabled
warning ("off", "Octave:broadcast");
% warning ("error", "Octave:broadcast"); % to turn it on uncomment this line. 
function cost = kmcost(Data,Centroids) % Cost function for kmeans
		cost = sum((Data-Centroids).^2)/size(Data,1)
function centroids = kmeans(Data,NoCentroids,Iters,tolerance?) % This is the outter function
		centroids = Data(randperm(size(Data,1),NoCentroids),:); % Initializing centroids as = random datapoints
function centroids = kmeansstep(Data,Centroids), 	% We'll assume data is in the same format as before,with each row being a datapoint
													% And each column being a variable in the datapoints
	% This is the assignment step
	[lowestcost,index]=min(sqrt(sum((perm(Data,[3,2,1])-Centroids).^2,2))); 
	index = permute(index,[3,2,1]);
	% Now this will need notes for later, first we make data from a M by X matrix, to a 1 by X by M matrix, then subtract from it
	% the centroids. Due to matrix broadcasting, this will result in data becoming a N by X by M matrix, where N is the number of
	% centroids, rows in centroid matrix etc, where data(n,:,:) for every n in N is the same. Centroids expand to N by X by M matrix
	% too, but one where centroids(:,:,m) for every m in M is the same. This then does elementwise subtraction on each producing a
	% N by X by M matrix , we'll call VDiff, where VDiff(n,:,m) is Data(m)-Centroids(n). This is a 1 by X by 1 array, a row vector
	% where each column is the difference in a component between Data(m) & Centroids(n). To get the eucledian difference from this
	% we square each of it, and then we sum  along columns to produce a N by 1 by M array, which we then apply sqrt on each point in
	% to get VDist, where VDist(n,:,m) is  a scalar that is the eucledian distance between Data(m) and Centroids(n). We then use min
	% to find the lowest value in each column (ie VDist(:,:,m)) and return its value & index. We then repermute index to restore it
	% to parity with data, giving us index where index(m) is the centroid to which Data(m,:) is assigned.

	% This is the moving centroid step
	for ex = size(Centroids); 	% This is the same as for ex = unique(index)
		filter = index==ex;		% This turns values other than ex to 0, and so creates a M by 1 matrix of 1s & 0s, with 
								% filter(m)=1 meaning Data(m,:) is assigned to Centroid(ex) and should be included
		Centroids(ex)=(filter'*data)/sum(filter) %This applies filter to data while summing the variables into position 
		% simultaneously while divind by the number of datapoints we used to get the average.
		% and thats centroid assignment.