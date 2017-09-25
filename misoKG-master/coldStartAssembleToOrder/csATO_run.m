% the input parameters
%b = [10 10 10 10 10 10 10 20];
%length = 10;
%seed = randi(100000000);

% Arrival rates for products
%arrivalratep1 = 3.6
%arrivalratep2 = 3
%arrivalratep3 = 2.4
%arrivalratep4 = 1.8
%arrivalratep5 = 1.2

% invocation of the simulator
[fn, FnVar] = csATO(b,length,seed,ar1,ar2,ar3,ar4,ar5,pr1,pr2,pr3,pr5,hc,apt1,apt5);
% [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = csATO(b,length,seed,arrivalratep1,arrivalratep2,arrivalratep3,arrivalratep4,arrivalratep5);
% ATO() returns more, ignore the others for now

% output the mean and variance of the profit after 20 days
formatSpec = 'fn=%4.8f\nFnVar=%4.8f\n';
fprintf(1,formatSpec,fn,FnVar) % 1 is for stdout

exit;