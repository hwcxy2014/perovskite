import coldstart_rosenbrock
import coldstart_ato
import miso_atoext
import miso_rosenbrock
import miso_lrMU
import miso_lrMU2
import miso_lrMU3
import miso_lrMU4
import miso_lrMU4150st

__author__ = 'jialeiwang'

def identify_problem(argv, bucket):
    benchmark_name = argv[0]
    decode = benchmark_name.split("_")
    if decode[0] == "coldstart":
        if decode[1] == "rb":
            problem_class = coldstart_rosenbrock.class_collection[benchmark_name]
        elif decode[1] == "ato":
            problem_class = coldstart_ato.class_collection[benchmark_name]
        else:
            raise ValueError("func name not recognized")
        if decode[2] == "hyper":
            problem = problem_class()
        elif decode[2] == "benchmark":
            replication_no = int(argv[1])
            obj_func_idx = int(argv[2])
            problem = problem_class(obj_func_idx, replication_no)
        else:
            raise ValueError("should only be benchmark or hyper")
    elif decode[0] == "miso":
        if decode[1] == "rb":
            problem_class = miso_rosenbrock.class_collection[benchmark_name]
            which_rb = int(argv[1])
            if decode[2] == "hyper":
                problem = problem_class(which_rb, bucket)
            else:
                replication_no = int(argv[2])
                problem = problem_class(replication_no, which_rb, bucket)
        elif decode[1] == "atoext":
            problem_class = miso_atoext.class_collection[benchmark_name]
            if decode[2] == "hyper":
                problem = problem_class(bucket)
            else:
                replication_no = int(argv[1])
                problem = problem_class(replication_no, bucket)
        elif decode[1] == "lrMU":
            problem_class = miso_lrMU.class_collection[benchmark_name]
            if decode[2] == "hyper":
                problem = problem_class(bucket)
            else:
                replication_no = int(argv[1])
                problem = problem_class(replication_no, bucket)
        elif decode[1] == "lrMU2":
            problem_class = miso_lrMU2.class_collection[benchmark_name]
            if decode[2] == "hyper":
                problem = problem_class(bucket)
            else:
                replication_no = int(argv[1])
                problem = problem_class(replication_no, bucket)
        elif decode[1] == "lrMU3":
            problem_class = miso_lrMU3.class_collection[benchmark_name]
            if decode[2] == "hyper":
                problem = problem_class(bucket)
            else:
                replication_no = int(argv[1])
                problem = problem_class(replication_no, bucket)
        elif decode[1] == "lrMU4":
            problem_class = miso_lrMU4.class_collection[benchmark_name]
            if decode[2] == "hyper":
                problem = problem_class(bucket)
            else:
                replication_no = int(argv[1])
                problem = problem_class(replication_no, bucket)
        elif decode[1] == "lrMU4150st":
            problem_class = miso_lrMU4150st.class_collection[benchmark_name]
            if decode[2] == "hyper":
                problem = problem_class(bucket)
            else:
                replication_no = int(argv[1])
                problem = problem_class(replication_no, bucket)
        else:
            raise ValueError("func name not recognized")
    else:
        raise ValueError("task name not recognized")
    return problem
