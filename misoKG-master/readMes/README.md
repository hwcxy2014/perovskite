# README #

### Steps for plugging in new objective functions
* Implement interface for objective function, refer to ``misoRosenbrock/rosenbrock.py`` for example
* Prepare data for experiments, refer to ``coldstart_gen_benchmark_data.py`` or ``coldstart_gen_hyper_data.py`` for example, note that
function evaluations are always a minimization problem when stored as data, to ensure consistency.
* Add new benchmark experiments, each experiment is considered a ``problem``, and define the configuration under ``problems/``
* Add problem identifier in ``problems/identifier.py``
* Write nbs scripts for running the problems under ``nbs/``

### Unittest suite
* We use ``nosetests`` to do the unittests, while I cannot figure out how to let nosetests read config automatically, I usually
do this command to perform tests: nosetests --nocapture -v /path/to/tests
* The option --nocapture allows me to print out values, and -v gives me more verbose outputs.
* For using features of nosetests, check out this post: http://www.metaklass.org/nose-making-your-python-tests-smell-better/

### Things to remember
* When I store data, the func evals are always in minimization form, e.g., ATO problems were maximization by default, so when you
store data, you need to flip sign of evals!
