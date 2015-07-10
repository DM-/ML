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
% performs regulization on a value. This can be tacked onto lin grad descnent, and log.
function regValue = reg(Value,lRate,rRate,m)
	regdValue = Value*(1-lRate*rRate/m);
end

%Normal Equation
function theta = nEq(x, y)
	theta = pinv(x' * x) * x' * y;
end

% Prepad with bias
function padded = prebias(m2b)
	padded = prepad(m2b,size(m2b,2)+1,1,2);
end