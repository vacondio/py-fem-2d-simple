#!/usr/bin/env python3

import numpy as np

# user input
Nx = 2      # number of subdivisions in x
Ny = 2      # number of subdivisions in y
Lx = 1.0    # x length of mesh rectangle
Ly = 1.0    # y length of mesh rectangle

# number of nodes (and number of basis functions)
nx = Nx+1
ny = Ny+1
n  = nx*ny

# number of elements
NE = 2*Nx*Ny

# number of non-vanishing "matrix elements" in the stiffness matrix
n_stiff = 9*NE

# represent the mesh using nodes and elements arrays
xmesh = np.linspace(0,Lx,nx)
ymesh = np.linspace(0,Ly,ny)
dx=xmesh[1]-xmesh[0]
dy=ymesh[1]-ymesh[0]

nodes    = np.array([(ymesh[j], xmesh[i]) for j in range(ny) for i in range(nx)])

elements = [([nodes[i],nodes[i+1],nodes[nx+i]]) for i in range(n-nx) if (i+1)%nx]
elements.extend([([nodes[i],nodes[i+1],nodes[i+1-nx]]) for i in range(nx,n-1) if (i+1)%nx])
elements = np.array(elements)
# N.B.: second half of the elements array contains the flipped elements

elements_idx = [(i,i+1,nx+i) for i in range(n-nx) if (i+1)%nx]
elements_idx.extend([(i,i+1,i+1-nx) for i in range(nx,n-1) if (i+1)%nx])
elements_idx = np.array(elements_idx)

# print results
# np.set_printoptions(legacy='1.25')
print("nodes:")
print(nodes)
print("\nelements:")
print(elements)
print("\nelements_idx:")
print(elements_idx)

