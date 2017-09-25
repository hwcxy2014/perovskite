import subprocess
import time					# used to measure time
from random import randint 	# becomes redundant if the random int is supplied externally

__author__ = 'matthiaspoloczek'

'''
Run the ATO Simulator from python.

To avoid overlong simulations, I invoke it as 'timeout 120s python ato_matlab.py', but you can
replace the timebound by whatever you deem reasonable.

The engine of Matlab does not work due to some undef symbol
The python-matlab-bridge does not run on the cluster because the GLibc is outdated.
Steve says he cannot update it, so I decided to do it the basic way.
'''

# Run the ATO simulator
# b_vector is currently a string, but you can adapt it to take whatever type of array you use
# simulation_length is a positive int that gives the length of the simulation
# random_seed should be a random int larger zero
# return the mean, the variance, and the elapsed time
def runATOsimulator(b_vector, simulation_length, random_seed):

	#TODO b_vector is a string (see my comment below) but of course we should pass the vector and then convert it approp.
	runcmd = b_vector+"length="+str(simulation_length)+";seed="+str(random_seed)+";run(\'ATO_run.m\');exit;"
	#runcmd = "exit;"
	# watch out for quotes and doublequotes they mess everything up
	#
	#timelimit = "120s"
	#print "Running ATO with timelimit " + timelimit + "\n"
	fn = -1.0
	FnVar = -1.0
	elapsed_time = 0.0

	try:
		start_time = time.time()
		# /usr/local/matlab/2015b/bin/matlab -nodisplay -nosplash -nodesktop -r "run('ATO_run.m');exit;"
		# https://www.mathworks.com/matlabcentral/answers/97204-how-can-i-pass-input-parameters-when-running-matlab-in-batch-mode-in-windows

		#stdout = "\n\n\nfn=58.65959236\nFnVar=10.51214922\n"
		stdout = subprocess.check_output(["/usr/local/matlab/2015b/bin/matlab", "-nodisplay", "-nojvm",
										  "-nosplash", "-nodesktop", "-r", runcmd])
		# , shell=True # is a risk
		elapsed_time = time.time() - start_time
		#print "stdout=\n"+stdout+"\n\n"

		posfn = stdout.find("fn=") + 3
		posFnVar = stdout.find("FnVar=") + 6
		if ((posfn > 2) and (posFnVar > 5)):
			posfnEnd = stdout.find("\n",posfn)
			posFnVarEnd = stdout.find("\n",posFnVar)
			fn = stdout[posfn:posfnEnd]
			FnVar = stdout[posFnVar:posFnVarEnd]
	except subprocess.CalledProcessError, e:
		elapsed_time = time.time() - start_time

	# elapsed_time is a reasonable cost
	# fn and FnVar are the results of interest
	#print "fn="+str(fn)+" , FnVar="+str(FnVar)+" , elapsed_time="+str(elapsed_time)
	# for presentation, can be removed

	return fn, FnVar, elapsed_time


# Illustration of how to invoke the simulator
#
def main():
	# Jialei, the following parameters are the simulation args
	b_vector = 'b=[19 17 14 20 16 13 17 15 ];' 	# the b vector
	simulation_length = 500 					# denoted by runlength in the matlab code
	random_seed = randint(1,10000000) 			# a random int that serves as seed for matlab

	fn, FnVar, elapsed_time = runATOsimulator(b_vector,simulation_length,random_seed)
	print "fn="+str(fn)+" , FnVar="+str(FnVar)+" , elapsed_time="+str(elapsed_time)

def estimateParametersOfIS():
	# Jialei, the following parameters are the simulation args
	b_vector = 'b=[19 17 14 20 16 13 17 15];' 	# the b vector
	#TODO this string should be a vector, since I do not know what you are using, I kept it like this
	simulation_length = 100 					# denoted by runlength in the matlab code
	random_seed = randint(1,10000000) 			# a random int that serves as seed for matlab
	#TODO For better and cheaper randomness, this int should be supplied externally from a PRG that is queried consecutively

	fn, FnVar, elapsed_time = runATOsimulator(b_vector,simulation_length,random_seed)
	print "fn="+str(fn)+" , FnVar="+str(FnVar)+" , elapsed_time="+str(elapsed_time)

main()