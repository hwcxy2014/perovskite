##################################
#
# Module calling XFOIL solver
#
##################################


import numpy as np
import os
import re

#TODO the "../dragAndLift" prefix should fix Remi's problem of relative paths. But there should be a better solution.

def write_conf_xfoil(mach,aoa):
    file = open("../dragAndLift/cfd_files/config_xfoil", "w")

    file.write("NACA 0012\nPLOP\nG\n\nOPER\n")
    file.write("mach %.5f \n" % mach)
    file.write("v 1e5\na 0.0\nP\n\n\nINIT\n")
    file.write("A %.5f \n" % aoa)
    file.write("PWRT 1\n../dragAndLift/cfd_files/XFOIL_results.txt\nY\n\nQUIT")
    file.write("")

    file.close()


def call_XFOIL(mach,aoa):

    # print 'Mach number :', mach
    # print 'AoA         :', aoa

    # write the configure file
    write_conf_xfoil(mach,aoa)
    # Submit job
    os.system("xfoil < ../dragAndLift/cfd_files/config_xfoil > /dev/null")
    # Parse results
    aoa, CL, CD, CDp, CM, Top_Xtr, Bot_Xtr = np.genfromtxt("../dragAndLift/cfd_files/XFOIL_results.txt", dtype = None, skip_header = 12)

    # Define the quantity of interest
    QOI = CD*1000.0

    return QOI , CL

def write_conf_su2(mach,aoa):
    max_iter = 10000
    # output_file = "./path"
    # path_geom = "./path"

    repl_mach  = "MACH_NUMBER= " + str(mach)
    repl_aoa   = "AOA= " + str(aoa)
    repl_iter  = "EXT_ITER= "+ str(max_iter)
    f = open('../dragAndLift/cfd_files/turb_NACA0012.cfg','r')
    out = open('../dragAndLift/cfd_files/config_su2','w')
    for line in f:
        line =re.sub('MACH_NUMBER= [0-9]*\.?[0-9]*',repl_mach, line)
        line =re.sub('AoA= [0-9]*\.?[0-9]*',repl_aoa, line)
        line =re.sub('EXT_ITER= [0-9]*',repl_iter, line)
        out.write(line)
    f.close()
    out.close()

def parse_result_su2(file_name):

    f = open(file_name,'r')
    for line in f:
        pass
    last = line
    f.close()

    splitlist = map(float,last.split(','))
    # Iteration,CLift,CDrag,CSideForce,CMx,CMy,CMz,CFx,CFy,CFz,CL_CD, \
    # Res_Flow_0,Res_Flow_1,Res_Flow_2,Res_Flow_3,Res_Flow_4,Res_Turb_0,\
    # Linear_Solver_Iterations,Time = splitlist

    # QOI = CL_CD

    return splitlist


def call_SU2(mach,aoa):

    # write the configure file
    write_conf_su2(mach,aoa)
    print "SU2 running ... be patient"
    print "   mach :", mach
    print "   aoa  :", aoa
    # Submit job
    # os.system("SU2_CFD cfd_files/config_su2")
    os.system("SU2_CFD ../dragAndLift/cfd_files/config_su2 > /dev/null")
    # Parse results
    Iteration,CLift,CDrag,CSideForce,CMx,CMy,CMz,CFx,CFy,CFz,CL_CD, \
    Res_Flow_0,Res_Flow_1,Res_Flow_2,Res_Flow_3,Res_Flow_4,Res_Turb_0, \
    Linear_Solver_Iterations,Time = parse_result_su2('../dragAndLift/cfd_files/history.plt')

    # Define quantity of interest
    QOI = CDrag*1000.0

    return QOI , CLift


if __name__ == "__main__":

    print "----------------------------\nCall XFOIL and SU2\n----------------------------"

    mach = 0.7
    aoa  = 1

    for i in range(0,5):
        QOI, constraint = call_XFOIL(mach,aoa)
        # print "Quantity of interest CL = ",QOI
        print "XFOIL: Iteration \t", i
    # map(float, l.split())
    print QOI, constraint

    print 'SU2'
    QOI, constraint = call_SU2(mach,aoa)
    print QOI , constraint