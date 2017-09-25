This instruction is tested on installing Xfoil 6.97. Please read READMEs in Xfoil first, as this instruction is based on the formal installation instructions with a few tweaks.

Step 0: we compiled Xfoil using gfortran and gcc, and all the following steps assume that you use the same compilers. Make sure the version is 4.8.X, as version 5.1.0 is tested to not work on Xfoil.

Step 1: Make some changes in two files:
        - open ./src/xoper.f and go to line 117, which was "IINPUT(I) = 2**31", change it to "IINPUT(I) = HUGE(0)"
        - open ./src/pplot.f and go to line 39, which was "LOGICAL ERROR, LGETFN", change it to "LOGICAL ERROR, LGETFN, LERR"

Step 1: Go to dir ./orrs and follow directions in ./orrs/README to modify osmap.f. We only tested single precsion setting.

Step 2: Open ./orrs/bin/Makefile, and do the following changes:
        - line 7, change to FC = gfortran
        - comment out line 14-17 since we do not use ifortran
        - save and in .orrs/bin/ do % make osgen and % make osmap.o

Step 3: Go to ./orrs/ and do % bin/osgen osmaps_ns.lst 

Step 4: Open ./plotlib/Makefile and do the following changes:
        - line 72, change it to "FC = gfortran"
        - line 73, change it to "CC = gcc"
        - comment out line 78-82 since we do not use double precision

Step 5: Open ./plotlib/config.make, and change the following lines:
        - line 54, change to "FC = gfortran"
        - line 55, change to "CC = gcc"

Step 6: Go to ./plotlib/ and do % make or % make libPlt.a

Step 7: Open ./bin/Makefile and do the following changes:
        - line 47, change to "FC = gfortran"
        - line 52, change to "CC = gcc"
        - comment out line 102-104 (not using ifort) and 111-116 (single precision)

Step 8: Go to ./bin and do % make xfoil, % make pplot and % make pxplot

Step 9: Edit you env variable to include Xfoil executable.

That should be it!
