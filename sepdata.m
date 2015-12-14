classA(1,:) = randn(1,100) .* 0.5 + 1.0;
classA(2,:) = randn(1,100) .* 0.5 + 0.5;
classB(1,:) = randn(1,100) .* 0.5 - 1.0;
classB(2,:) = randn(1,100) .* 0.5 + 0.0;
patterns = [classA, classB];
targets(1,1:100)=ones(1,100);
targets(1,101:200)=ones(1,100).*-1;
permutation = randperm(200);
patterns = patterns(:, permute)';
targets = targets(:, permute)';