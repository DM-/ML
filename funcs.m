function a = sigm(z),
	%Used for logHypothesis
	a=1./(1+exp(z.*-1));
end;

function a = invsigm(z),
	a=sigm(z).*(1.-sigm(z));
end;

function a = self(z);
	a=z;
end;

function a = tin(z);
	a= 2 ./ (1+exp(-z)) - 1;
end;

function a = invtin(z);
	a= ((1 + tin(z)) .* (1 - tin(z))) * 0.5;
end;