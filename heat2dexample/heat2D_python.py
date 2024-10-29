# -*- coding: utf-8 -*-
"""
# FILE: heat2D_python.py
# DESCRIPTIONS:
# HEAT2D Example - Serial Python Version
# This example is based on a two-dimensional heat
# equation.  The initial temperature is computed to be
# high in the middle of the domain and zero at the boundaries.  The
# boundaries are held at zero throughout the simulation.  During the
# time-stepping, an array containing two domains is used; these domains
# alternate between old data and new data.
"""
#****************************************************************************/
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

#*****************************************************************************
#  function inidat
#*****************************************************************************/
def inidat(nx,ny,u):
    for ix in range(nx//3,2*nx//3):
        for iy in range(ny//3,2*ny//3):
#            u[0,ix,iy] = ix * (nx - ix - 1) * iy * (ny - iy - 1)
            u[0,ix,iy] = 1

#**************************************************************************
# function prtdat
#**************************************************************************/
def prtdat(nx,ny,u1,fnam):
    fp = open(fnam, 'w')
    for ix in range(nx):
        for iy in range(ny):
            print("{:8.1f}".format(u1[ix,iy]),end=' ',file=fp)
        fp.write("\n")
    fp.close()

#**************************************************************************
# function pltdat
#**************************************************************************/
def pltdat(u1,range,fnam):
    plt.imsave(fnam,u1,cmap='hot',vmin=range[0],vmax=range[1])
#    plt.figure()
#    plt.set_cmap('hot')
#    plt.imshow(u1,vmin=range[0],vmax=range[1])
#    plt.show()

#**************************************************************************
# function update
#****************************************************************************/
def update(nx, ny, u1, u2):
    temp = 1-2*(Cx+Cy)
    for ix in range(1,nx-1):
        for iy in range(1,ny-1):
            u2[ix,iy] = u1[ix,iy]*temp + Cx * (u1[ix+1,iy] + u1[ix-1,iy]) + Cy * (u1[ix,iy+1] + u1[ix,iy-1])

#**************************************************************************

Cx = 0.1          # blend factor in heat equation
Cy = 0.1          # blend factor in heat equation

#************************* main code *******************************/
def main( PROGNAME, STEPS, NXPROB, NYPROB ):
# Initialize grid
    u = np.zeros((2,NXPROB,NYPROB))        # array for grid
    inidat(NXPROB, NYPROB, u)
#    prtdat(NXPROB, NYPROB, u[0], "initial.dat")
    plotrange = (u.min(),u.max())
    pltdat(u[0], plotrange, "initial.png")

# Begin doing STEPS iterations.  
    initial = time.time()
    iz = 0;
    # Now call update to update the value of grid points
    for it in range(STEPS):
        update(NXPROB,NYPROB,u[iz],u[1-iz]);
        iz = 1 - iz

    final = time.time()
    print("{}: X= {:d}  Y= {:d}  Steps= {:d} time: {:8.6f} s".format(PROGNAME, NXPROB,NYPROB,STEPS,final-initial))

# Write final output, call X graph and finalize MPI
#    prtdat(NXPROB, NYPROB, u[iz], "final.dat")
    pltdat(u[iz],plotrange,"final.png")

#************************* input code *******************************/
if __name__ == '__main__':
    if int(len(sys.argv)) == 4:
        PROGNAME = sys.argv[0]
        STEPS = int(sys.argv[1])
        NXPROB = int(sys.argv[2])
        NYPROB = int(sys.argv[3])
        main(PROGNAME, STEPS, NXPROB, NYPROB)
    else:
        print("Usage: {} <ITERATIONS> <XDIM> <YDIM>".format(sys.argv[0]))

