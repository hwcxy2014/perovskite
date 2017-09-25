##################################
#
# Module calling XFOIL solver
#
##################################


import numpy as np
import os
import re


def write_conf_xfoil(mach, aoa):
    file = open("cfd_files/config_xfoil", "w")

    file.write("NACA 0012\nPLOP\nG\n\nOPER\n")
    file.write("mach %.5f \n" % mach)
    file.write("v 1e5\na 0.0\nP\n\n\nINIT\n")
    file.write("A %.5f \n" % aoa)
    file.write("PWRT 1\ncfd_files/XFOIL_results.txt\nY\n\nQUIT")
    file.write("")

    file.close()


def call_XFOIL(mach, aoa):
    # write the configure file
    write_conf_xfoil(mach, aoa)
    # Submit job
    os.system("xfoil < cfd_files/config_xfoil > /dev/null")
    # Parse results
    aoa, CL, CD, CDp, CM, Top_Xtr, Bot_Xtr = np.genfromtxt("cfd_files/XFOIL_results.txt", dtype=None, skip_header=12)

    # Define the quantity of interest
    QOI = CL / CD

    return QOI


def write_conf_su2(mach, aoa):
    max_iter = 5000
    # output_file = "./path"
    # path_geom = "./path"

    repl_mach = "MACH_NUMBER= " + str(mach)
    repl_aoa = "AOA= " + str(aoa)
    repl_iter = "EXT_ITER= " + str(max_iter)
    f = open('cfd_files/turb_NACA0012.cfg', 'r')
    out = open('cfd_files/config_su2', 'w')
    for line in f:
        line = re.sub('MACH_NUMBER= [0-9]*\.?[0-9]*', repl_mach, line)
        line = re.sub('AoA= [0-9]*\.?[0-9]*', repl_aoa, line)
        line = re.sub('EXT_ITER= [0-9]*', repl_iter, line)
        out.write(line)
    f.close()
    out.close()


def parse_result_su2(file_name):
    f = open(file_name, 'r')
    for line in f:
        pass
    last = line
    f.close()

    splitlist = map(float, last.split(','))
    # Iteration,CLift,CDrag,CSideForce,CMx,CMy,CMz,CFx,CFy,CFz,CL_CD, \
    # Res_Flow_0,Res_Flow_1,Res_Flow_2,Res_Flow_3,Res_Flow_4,Res_Turb_0,\
    # Linear_Solver_Iterations,Time = splitlist

    # QOI = CL_CD

    return splitlist


def call_SU2(mach, aoa):
    # write the configure file
    write_conf_su2(mach, aoa)
    # Submit job
    os.system("SU2_CFD cfd_files/config_su2")
    # os.system("SU2_CFD config_su2 > /dev/null")
    # Parse results
    Iteration, CLift, CDrag, CSideForce, CMx, CMy, CMz, CFx, CFy, CFz, CL_CD, \
    Res_Flow_0, Res_Flow_1, Res_Flow_2, Res_Flow_3, Res_Flow_4, Res_Turb_0, \
    Linear_Solver_Iterations, Time = parse_result_su2('cfd_files/history.plt')

    # Define quantity of interest
    QOI = CL_CD

    return QOI


if __name__ == "__main__":
    print "----------------------------\nCall XFOIL and SU2\n----------------------------"

    mach = 0.3
    aoa = 8

    # for i in range(0,5):
    QOI = call_XFOIL(mach, aoa)
    # print "Quantity of interest CL = ",QOI
    # print "XFOIL: Iteration {0}, QOI {1}".format(i, QOI)
    print "XFOIL QOI {0}".format(QOI)
# map(float, l.split())

# print 'SU2'
# QOI = call_SU2(mach,aoa)
# print QOI
