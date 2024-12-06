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

nodes = np.array([(ymesh[j], xmesh[i]) for j in range(ny) for i in range(nx)])
# contains n = nx*ny couples

elements = [([nodes[i],nodes[i+1],nodes[nx+i]]) for i in range(n-nx) if (i+1)%nx]
elements.extend([([nodes[i],nodes[i+1],nodes[i+1-nx]]) for i in range(nx,n-1) if (i+1)%nx])
elements = np.array(elements)
# contains NE = 2*Nx*Ny elements
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
#    (x0,y0) = 0 ______ 1 = (x1,y1)
#                |   /| 
#                |  /|| 
#                | /||| 
#    (x2,y2) = 2 |/|||| 
# 
# Flipped elements are indexed in this way instead:
# 
#                _____  2 = (x2,y2)
#                ||||/|     
#                |||/ |     
#                ||/  |     
#    (x0,y0) = 0 |/___| 1 = (x1,y1)
#
# Therefore we have for a flipped element that:
#
#    1 -> 0,  0 -> 1,  2 -> 2
#
#  [[ 00, 01, 02 ]          [[ 11, 10, 12 ]
#   [ 10, 11, 12 ]    ->     [ 01, 00, 02 ]
#   [ 20, 21, 22 ]]          [ 21, 20, 22 ]]
#
# Let us now write down the formula for the computation of the local stiffness,
# assuming x0 = 0 and y0 = 0:
#
# D =  [ x2-x1, x0-x2, x1-x0 ] = [ -dx,   0, dx ]
#      [ y2-y1, y0-y2, y1-y0 ]   [  dy, -dy,  0 ]
#
# Finally we compute the local stiffness with:
#
#        D^T x D
# A = --------------
#      4*elem_area
#

print("\n\nlocal stiffness matrix:")
print(local_stiffn())
print("\nflipped local stiffness matrix:")
print(local_stiffn(flip=True))

def stiffn(nodes_=0, elements_idx_=0, large=1e05, apply_dirichlet_cs=True, return_banded=False):
    # infer data from given mesh, to be done later
    n_data   = n_stiffn
    n_data_h = int(n_data/2)
    NE_h     = int(NE/2)

    # allocate extra space (4*nx) for boundary conditions
    rows = np.zeros(n_data + 4*nx)
    cols = np.zeros(n_data + 4*nx)
    data = np.zeros(n_data + 4*nx)
    
    rows[:n_data] = np.repeat(elements_idx,3).flatten() # add extra flatten 'cos you never know
    cols[:n_data] = np.tile(elements_idx,3).flatten()
    
    data[0:n_data_h] = np.tile(local_stiffn().flatten(),NE_h)
    data[n_data_h:n_data] = np.tile(local_stiffn(flip=True).flatten(),NE_h)

    # stiffn_mat = coo_matrix((data,(rows,cols)), shape=(n,n))
    # stiffn_mat.sum_duplicates() # yes, there are some

    if (apply_dirichlet_cs):
        # impose boundary conditions
        # M = abs(stiffn_mat).max() * large
        top_idx        = np.arange(0     , nx    )
        bottom_idx     = np.arange(n-nx  , n     )
        left_idx       = np.arange(0     , n-nx+1, nx)
        right_idx      = np.arange(nx-1  , n     , nx)
        boundaries_idx = np.append(top_idx, [bottom_idx,left_idx,right_idx])
        
        rows[n_data:] = boundaries_idx
        cols[n_data:] = boundaries_idx
        
        M = abs(data).max() * large
        data[n_data:] = M

    if (return_banded):
        rows = rows + nx - cols
        stiffn_mat = coo_matrix((data,(rows,cols)), shape=(2*nx+1,n))
    else:
        stiffn_mat = coo_matrix((data,(rows,cols)), shape=(n,n))
        
    stiffn_mat.sum_duplicates() # yes, there are some    
    
    return stiffn_mat, rows, cols, data
    # return stiff_mat
                                                           
stiffn_mat, rows, cols, data = stiffn(apply_dirichlet_cs=False)
stiffn_dense_mat = stiffn_mat.todense()

stiffn_bnd_mat, rows, cols, data = stiffn(apply_dirichlet_cs=False,return_banded=True)
stiffn_bnd_dense_mat = stiffn_bnd_mat.todense()

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

print("\n\n stiffness matrix in banded format:")
print(stiffn_bnd_dense_mat)
# One could check that stiffn_dense_mat is positive definite, but it seems to me
# it is since it is clearly diagonally dominant with a positive diagonal.  And
# of course it is worth checking that the determinant is different from zero.
    

#===============================================================================
# 3. CONSTANTS VECTOR GENERATION STEP (BASIS PROJECTION)
# ==============================================================================
#
# Each basis function is non-vanishing over a the surface of a hexagon made of 6
# elements.  Therefore, we need to evaluate the integral of \int dx f(x)*v_i(x)
# over each hexagon indexed by i.  To keep things simple, we will evaluate the f
# over the mesh that we have already generated, and use the available values of
# f to compute the integral.  This means that, for each hexagon we have, only
# one value of the approximated integrand does not vanish, since v_i(x_j) = 0
# for j/=i.  Therefore, f(x_i)*v_i(x_i) is the only non-vanishing value of the
# integrand, and all the neighboring mesh nodes yield vanishing contributions.
# With a linear approximation of f(x)v_i(x), we find that the integral over each
# element of the hexagon is simply the volume of the tetrahedron having the
# element as base and f(x_i)v_i(x_i) as height, that being
#
# 1/3 * 1/2 dx dh * f(x_i) v_i(x_i)
#
# Each of these volumes is to be multiplied by 6 to get the integral over the
# whole hexagon.

def g(p):
    sigma_x = 0.1
    sigma_y = 0.1
    x0      = 0.5
    y0      = 0.5
    N       = 1.0
    return N * np.exp(-((p[...,0]-x0)**2/(2.0*sigma_x) + (p[...,1]-y0)**2/2.0*sigma_y))

def fv_int(f,nodes_):
    const_int = 6.0 / 3.0 * 0.5 * dx*dy
    fv_vec  = const_int * np.array(f(nodes_))

    # set boundary conditions
    fv_vec[0:    nx    ]     = 0.0
    fv_vec[n-nx: n     ]     = 0.0
    fv_vec[0:    n-nx+1: nx] = 0.0
    fv_vec[nx-1: n:      nx] = 0.0

    return fv_vec
    
b_vec = np.array(fv_int(g,nodes))

print("\n\nf(x):")
print(g(nodes))
print("\n\nb_vec:")
print(b_vec)

#===============================================================================
# 4. LINEAR SYSTEM SOLUTION
#===============================================================================


#===============================================================================
# 5. EXAMPLE + PLOT OF THE RESULTS
#===============================================================================
