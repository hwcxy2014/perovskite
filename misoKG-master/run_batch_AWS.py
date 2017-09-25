import subprocess

'''
Run algos as batches on AWS

Optional parameters for lrMU:
    "miso_lrMU_benchmark_ego":
    "miso_lrMU_benchmark_mkg":
    "miso_lrMU_benchmark_mkgcandpts":
    "miso_lrMU_benchmark_pes":
    "miso_lrMU_benchmark_mei":
'''


script_to_call = "./run_pes.py"  #"./run_mei.py" #"./run_mkgcandpts.py" #"./run_mkg.py" #"./run_ego.py"
benchmark = "miso_lrMU4_benchmark_pes" #"miso_lrMU4_benchmark_mei" #"miso_lrMU4_benchmark_mkgcandpts"
### "miso_lrMU3_benchmark_pes" #"miso_lrMU3_benchmark_mkgcandpts" #"miso_lrMU3_benchmark_mei"
### "miso_lrMU2_benchmark_mei" #"miso_lrMU2_benchmark_pes"
### "miso_lrMU_benchmark_mkgcandpts" #"miso_lrMU_benchmark_mei" #"miso_lrMU_benchmark_mkg" #"miso_lrMU_benchmark_ego" #"miso_lrMU_benchmark_pes"

for repl_no in range(60,61): #xrange(80,100): #range(65,80) + range(85,100):
    print 'Starting '+script_to_call+' on '+benchmark+' repl '+str(repl_no)
    try:
        subprocess.check_call(["python", script_to_call, benchmark, str(repl_no)])
    except subprocess.CalledProcessError, e:
        stdout = '\n\n\n Starting {0} on {1} repl {2}\n\n{3}'.format(script_to_call, benchmark, str(repl_no),
                                                                     str(e.output))
        with open('run_batch_AWS_log.txt', 'a') as errorlog:
            errorlog.write(stdout)

