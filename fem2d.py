#!/usr/bin/env python3

import numpy as np
from scipy.sparse import coo_matrix
from scipy.linalg import solve_banded

#===============================================================================
# 1. MESH GENERATION STEP
#===============================================================================

class TriangularMesh2D:
    """
    A simple triangular 2D mesh class.

    Attributes:
        Nx : number of subdivisions in x
        Ny : number of subdivisions in y
        Lx : x length of the mesh rectangle
        Ly : y length of the mesh rectangle

    Methods:
        densify : densify mesh in a mathematically controlled way (helps
                  convergence of FEM)
    """
    def __init__(self, Nx, Ny, Lx, Ly):

        # number of nodes (and number of basis functions)
        nx = Nx+1
        ny = Ny+1
        n  = nx*ny
        
        # number of elements
        NE = 2*Nx*Ny

        # build x and y meshes
        xmesh = np.linspace(0,Lx,nx)
        ymesh = np.linspace(0,Ly,ny)
        
        dx = xmesh[1]-xmesh[0]
        dy = ymesh[1]-ymesh[0]

        # represent the mesh using nodes and elements arrays
        nodes = np.array([(xmesh[i], ymesh[j])
                          for j in range(ny) for i in range(nx)])

        NE_h = int(NE/2)
        elements_idx = np.zeros([NE,3], dtype=int)
        elements_idx[:NE_h] = np.array([(i,i+1,nx+i)
                                        for i in range(n-nx) if (i+1)%nx])
        elements_idx[NE_h:] = np.array([(i,i+1,i+1-nx)
                                        for i in range(nx,n-1) if (i+1)%nx])
        # N.B.: second half of the elements array contains the flipped elements,
        # according to the following layout: first half of the elements is built
        # like so:
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
        
        # assign attributes to self
        self._Nx = Nx
        self._Ny = Ny
        self._Lx = Lx
        self._Ly = Ly
        self._nx = nx
        self._ny = ny
        self._n  = n
        self._NE = NE
        self._xmesh = xmesh
        self._ymesh = ymesh
        self._dx = dx
        self._dy = dy
        self._nodes = nodes
        self._elements_idx = elements_idx

    # getters
    @property
    def Nx(self):
        return self._Nx
    @property
    def Ny(self):
        return self._Ny
    @property
    def Lx(self):
        return self._Lx
    @property
    def Ly(self):
        return self._Ly
    @property
    def nx(self):
        return self._nx
    @property
    def ny(self):
        return self._ny
    @property
    def n(self):
        return self._n
    @property
    def NE(self):
        return self._NE
    @property
    def xmesh(self):
        return self._xmesh
    @property
    def ymesh(self):
        return self._ymesh
    @property
    def dx(self):
        return self._dx
    @property
    def dy(self):
        return self._dy
    @property
    def nodes(self):
        return self._nodes
    @property
    def elements_idx(self):
        return self._elements_idx


#===============================================================================
# 2. STIFNESS MATRIX GENERATION STEP
#===============================================================================

def local_stiffn(mesh,flip=False):
    dx = mesh.dx
    dy = mesh.dy
    d_mat = np.array([[-dx,   0, dx],
                      [ dy, -dy,  0]])
                     
    ls_mat  = d_mat.T@d_mat/(2.0*dx*dy)
    lsf_mat = np.array([[ls_mat[1,1],ls_mat[1,0],ls_mat[1,2]], 
                        [ls_mat[0,1],ls_mat[0,0],ls_mat[0,2]], 
                        [ls_mat[2,1],ls_mat[2,0],ls_mat[2,2]]])
    if flip:
        return lsf_mat
    else:
        return ls_mat
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


def stiffn(mesh, large=1e05, apply_dirichlet_cs=True, return_banded=True):
    NE     = mesh.NE
    n_data = 9*NE
    n      = mesh.n
    nx     = mesh.nx

    # allocate extra space (4*nx) for boundary conditions
    rows = np.zeros(n_data + 4*nx, dtype=int)
    cols = np.zeros(n_data + 4*nx, dtype=int)
    data = np.zeros(n_data + 4*nx, dtype=np.float64)
    
    rows[:n_data] = np.repeat(mesh.elements_idx,3)
    cols[:n_data] =   np.tile(mesh.elements_idx,3).flatten()

    NE_h     = int(NE/2)
    n_data_h = int(n_data/2)
    ls_mat  = local_stiffn(mesh)
    lsf_mat = local_stiffn(mesh, flip=True)
    data[0:n_data_h]      = np.tile( ls_mat.flatten(), NE_h)
    data[n_data_h:n_data] = np.tile(lsf_mat.flatten(), NE_h)

    if (apply_dirichlet_cs):
        # impose boundary conditions
        top_idx        = np.arange(0     , nx        , dtype=int)
        bottom_idx     = np.arange(n-nx  , n         , dtype=int)
        left_idx       = np.arange(0     , n-nx+1, nx, dtype=int)
        right_idx      = np.arange(nx-1  , n     , nx, dtype=int)
        boundaries_idx = np.concatenate([top_idx,bottom_idx,left_idx,right_idx])
        
        rows[n_data:] = boundaries_idx
        cols[n_data:] = boundaries_idx
        
        M = abs(data).max() * large
        data[n_data:] = M

    if (return_banded):
        rows = rows + nx - cols
        stiffn_mat = coo_matrix((data,(rows,cols)), shape=(2*nx+1,n))
    else:
        stiffn_mat = coo_matrix((data,(rows,cols)), shape=(n,n))
        
    stiffn_mat.sum_duplicates()
    # return stiffn_mat, rows, cols, data
    return stiffn_mat


#===============================================================================
# 3. CONSTANTS VECTOR GENERATION STEP (PROJECTION ONTO BASIS)
# ==============================================================================

def fv_int(mesh, f):
    dx, dy, nodes = mesh.dx, mesh.dy, mesh.nodes
    const_int = 6.0 / 3.0 * 0.5 * mesh.dx*mesh.dy
    fv_vec  = const_int * np.array(f(nodes))

    # set boundary conditions
    n, nx = mesh.n, mesh.nx
    fv_vec[0:   nx]         = 0.0
    fv_vec[n-nx:n ]         = 0.0
    fv_vec[0:   n-nx+1: nx] = 0.0
    fv_vec[nx-1:n:      nx] = 0.0

    return fv_vec
    # Each basis function is non-vanishing over a the surface of a hexagon made
    # of 6 elements.  Therefore, we need to evaluate the integral of \int dx
    # f(x)*v_i(x) over each hexagon indexed by i.  To keep things simple, we
    # will evaluate the f over the mesh that we have already generated, and use
    # the available values of f to compute the integral.  This means that, for
    # each hexagon we have, only one value of the approximated integrand does
    # not vanish, since v_i(x_j) = 0 for j/=i.  Therefore, f(x_i)*v_i(x_i) is
    # the only non-vanishing value of the integrand, and all the neighboring
    # mesh nodes yield vanishing contributions.  With a linear approximation of
    # f(x)v_i(x), we find that the integral over each element of the hexagon is
    # simply the volume of the tetrahedron having the element as base and
    # f(x_i)v_i(x_i) as height, that being
    #
    # 1/3 * 1/2 dx dh * f(x_i) v_i(x_i)
    #
    # Each of these volumes is to be multiplied by 6 to get the integral over
    # the whole hexagon.
    

if __name__ == "__main__":
    import sys  # in case we want to use command line arguments
    
    # user input
    Nx = 100    # number of subdivisions in x
    Ny = 100    # number of subdivisions in y
    Lx = 1.0    # x length of mesh rectangle
    Ly = 1.0    # y length of mesh rectangle

    # instantiate mesh
    fem_mesh = TriangularMesh2D(Nx, Ny, Lx, Ly)

    # define a 2D gaussian function, p = [(x0,y0), (x1,y1), ...]
    def g(p):
        sigma_x = 0.01
        sigma_y = 0.01
        x0      = 0.5
        y0      = 0.5
        N       = 1.0
        return N * np.exp(-((p[...,0]-x0)**2/(2.0*sigma_x) +
                            (p[...,1]-y0)**2/(2.0*sigma_y)))

    
    #===========================================================================
    # 4. LINEAR SYSTEM SOLUTION
    #===========================================================================
    #
    # We can solve the linear system in two ways:
    #
    # - by using the general numpy solver -- numpy.linalg.solve -- with the full
    #   n*n coefficients matrix A_mat foo foo foo foo;
    #
    # - by using the scipy solver specific to banded matrices --
    #   scipy.linalg.solve_banded -- with the A_mat_bnd matrix in banded format.
    #
    # Let us compare the two methods, and time them too!
         
    from time import time
    
    A_mat     = stiffn(fem_mesh, apply_dirichlet_cs=True, return_banded=False)
    A_mat_bnd = stiffn(fem_mesh, apply_dirichlet_cs=True, return_banded=True )
    b_vec     = fv_int(fem_mesh, g)

    # these should be moved inside stiffn()
    A_mat     = A_mat.todense()
    A_mat_bnd = A_mat_bnd.todense()
    
    start = time(); x1 = np.linalg.solve(A_mat, b_vec); end = time();
    print("\n          time to solution of np.linalg.solve : %f s" % (end - start))

    nx = fem_mesh.nx
    start = time(); x2 = solve_banded((nx,nx), A_mat_bnd, b_vec); end = time();
    print(  "time to solution of scipy.linalg.solve_banded : %f s" % (end - start))
    
    print("\nx1:")
    print(x1)
    print("\nx2:")
    print(x2)
    
    if np.allclose(x1, x2, 1e-64, 1e-15):
        print("\nnp.linalg.solve(A_mat, b_vec) yielded the same result as\n"
              "scipy.linalg.solve_banded(A_mat, b_vec), hooray!")
    else:
        print("\nnp.linalg.solve(A_mat, b_vec) did not yield the same result as\n"
              "scipy.linalg.solve_banded(A_mat, b_vec), alas!")
    
    #===========================================================================
    # 5. PLOT THE RESULTS
    #===========================================================================
    
    import matplotlib.pyplot as plt

    # prepare (x,y) grid
    nx, ny, nodes = fem_mesh.nx, fem_mesh.ny, fem_mesh.nodes
    X = nodes[:,0].reshape(nx,ny)
    Y = nodes[:,1].reshape(nx,ny)
    
    fig = plt.figure()
    
    # first subplot: g(x, y), gaussian function, inhomogeneity of the Poisson
    # equation
    Z = g(nodes).reshape(nx,ny)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_wireframe(X, Y, Z, cstride=5, rstride=5)
    
    # second subplot: x2, solution of the linear system
    Z = x2.reshape(nx,ny)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_wireframe(X, Y, Z, cstride=5, rstride=5)
    
    # # third subplot: x1, slow solution of the linear system 
    # ax = fig.add_subplot(1, 3, 3, projection='3d')
    # Z = x1.reshape(nx,ny)
    # ax.plot_wireframe(X, Y, Z, cstride=5, rstride=5)
    
    plt.show()    


    
