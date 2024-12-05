#!/usr/bin/env python3

import numpy as np
from scipy.sparse import coo_matrix

# user input
Nx = 2      # number of subdivisions in x
Ny = 2      # number of subdivisions in y
Lx = 1.0    # x length of mesh rectangle
Ly = 1.0    # y length of mesh rectangle

#===============================================================================
# 1. MESH GENERATION STEP
#===============================================================================

# number of nodes (and number of basis functions)
nx = Nx+1
ny = Ny+1
n  = nx*ny

# number of elements
NE = 2*Nx*Ny

# number of non-vanishing "matrix elements" in the stiffness matrix
n_stiffn = 9*NE

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

# First half of the elements is built like so:
# 
#        i _____ i+1
#          |   /|
#          |  / |
#          | /  |
#     nx+1 |/___|
# 
# Second half of the elements is made of flipped triangles:
# 
#          _____ i+1-nx
#          |   /|
#          |  / |
#          | /  |
#        i |/___|i+1
#
# We will refer to the latter element as a flipped element.

# print results
# np.set_printoptions(legacy='1.25')
print("nodes:")
print(nodes)
print("\nelements:")
print(elements)
print("\nelements_idx:")
print(elements_idx)

#===============================================================================
# 2. STIFNESS MATRIX GENERATION STEP
#===============================================================================

def local_stiffn(elements=0,flip=False):
    d_mat = np.array([[-dx,   0, dx],
                      [ dy, -dy,  0]])
                     
    ls_mat  = d_mat.T@d_mat/(2.0*dx*dy)
    lsf_mat = np.array([[ls_mat[1,1],ls_mat[1,0],ls_mat[1,2]], 
                        [ls_mat[0,1],ls_mat[0,0],ls_mat[0,2]], 
                        [ls_mat[2,1],ls_mat[2,0],ls_mat[2,2]]])
    if flip:
        return ls_mat
        # return lsf_mat
    else:
        return lsf_mat
        # return ls_mat

# The indexing of the local stiffness matrix is similar to that seen earlier
# when building the elements, except now i=0.  Non-flipped element is indexed
# like so:
# 
#        0 _____ 1
#          |   /|
#          |  / |
#          | /  |
#        2 |/___|
# 
# Flipped elements are indexed in this way instead:
# 
#          _____ 2
#          |   /|
#          |  / |
#          | /  |
#        0 |/___|1
#
# Therefore we have for a flipped element that:
#
#    1 -> 0,  0 -> 1,  2 -> 2
#
#  [[ 00, 01, 02 ]          [[ 11, 10, 12 ]
#   [ 10, 11, 12 ]    ->     [ 01, 00, 02 ]
#   [ 20, 21, 22 ]]          [ 21, 20, 22 ]]
    
print("\n\nlocal stiffness matrix:")
print(local_stiffn())
print("\nflipped local stiffness matrix:")
print(local_stiffn(flip=True))

def stiffn(nodes_=0, elements_idx_=0, large=1e05):
    # infer data from given mesh, to be done later
    rows = np.repeat(elements_idx,3).flatten() # add extra flatten 'cos you never know
    cols = np.tile(elements_idx,3).flatten()
    
    n_data_h = int(n_stiffn/2)
    NE_h = int(NE/2)
    data = np.zeros(n_stiffn)
    data[0:n_data_h] = np.tile(local_stiffn().flatten(),NE_h)
    data[n_data_h:n_stiffn] = np.tile(local_stiffn(flip=True).flatten(),NE_h)

    stiffn_mat = coo_matrix((data,(rows,cols)), shape=(n,n))
    stiffn_mat.sum_duplicates() # yes, there are some

    # impose boundary condition
    M = abs(stiffn_mat).max() * large
    # ... (find the indices)
    
    
    return stiffn_mat, rows, cols, data
    # return stiff_mat
                                                           
stiffn_mat, rows, cols, data = stiffn()
stiffn_dense_mat = stiffn_mat.todense()

print("\nrows:")
print(rows)
print("\ncols:")
print(cols)
print("\nflattened data:")
print(data)
print("\nstiffness matrix in sparse format:")
print(stiffn_mat)
print("\nstiffness matrix in dense format:")
print(stiffn_dense_mat)
if np.ma.allequal(stiffn_dense_mat,stiffn_dense_mat.T):
    print("stiffness matrix is symmetric, hooray!")
else:
    print("stiffness matrix is not symmetric, alas!")
# One could check that stiffn_dense_mat is positive definite, but it seems to me
# it is since it is clearly diagonally dominant with a positive diagonal. And of
# course it is worth checking that the determinant is different from zero.
    

#===============================================================================
# 3. CONSTANTS VECTOR GENERATION STEP (BASIS PROJECTION)
#===============================================================================


#===============================================================================
# 4. LINEAR SYSTEM SOLUTION
#===============================================================================


#===============================================================================
# 5. EXAMPLE + PLOT OF THE RESULTS
#===============================================================================
