% Runs all the tests.
function ok = TestAll(run_long_tests, run_interactive_tests)
if (nargin < 1), run_long_tests = 0; end
if (nargin < 2), run_interactive_tests = 0; end

ok = 1;

% util/test
ok = TestLogPlusExpSigned() && ok;

% belief/test
ok = TestPowExpCov() && ok;
ok = TestBayesUpdate1() && ok;
ok = TestBayesUpdateBlock() && ok;
ok = TestNoninformativeBayesUpdate() && ok;
ok = Test1DBayesUpdate() && ok;
ok = TestBayesUpdate2() && ok;
ok = TestBraninBayesUpdate() && ok;

% ckg/test
ok = TestCorrelatedKG() && ok;
ok = TestEmaxAffine() && ok;
ok = TestLogEmaxAffine() && ok;
ok = TestTimeCKG() && ok;

% bgo/test
ok = TestEGO() && ok;
ok = TestEGOSKO() && ok;
ok = TestEmpiricalVariogram() && ok;
ok = TestEstimateHyperparameters() && ok;


if (ok)
	disp('All tests OK');
else
	disp('Some tests FAILED');
end
