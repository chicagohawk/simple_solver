# Illustration of staggered grid (including ghost faces):
#
#          u_x[0,0]           u_x[1,0]           u_x[2,0]
#
#  u_y[0,0]   ----- u_y[1,0] --------- u_y[2,0] ---------
#             |                  |                  |
#             |                  |                  |
#          u_x[0,1]  p[0,0]   u_x[1,1]  p[1,0]   u_x[2,1]
#             |                  |                  |
#             |                  |                  |
#  u_y[0,1]   |---- u_y[1,1] ----+---- u_y[2,1] ----+----
#             |                  |                  |
#             |                  |                  |
#          u_x[0,2]  p[0,1]   u_x[1,2]  p[1,1]   u_x[2,2]
#             |                  |                  |
#
#                     ...                ...
#
#             |                  |                  |
#          u_x[0,N] p[0,N-1]  u_x[1,N] p[1,N-1]  u_x[2,N]
#             |                  |                  |
#             |                  |                  |
#  u_y[0,N]   ----- u_y[1,N] --------- u_y[2,N] ---------
#                                                        
#         u_x[0,N+1]         u_x[1,N+1]         u_x[2,N+1]

from pdb import set_trace
import sys
sys.path.append('../..')
from numpad import *
import unittest

class TestMain(unittest.TestCase):
    def Setup(self):
        xs = np.linspace(0,1,N)
        ys = xs.copy()
        self.u_x, self.u_y = np.meshgrid(xs, ys)
        self.p = self.u_x.copy()
        

def make_array(u):
    u_x = zeros([N+1, N+2])
    u_y = zeros([N+2, N+1])
    u_x[1:-1,1:-1] = u[:N*(N-1)].reshape([N-1,N])
    u_y[1:-1,1:-1] = u[N*(N-1):].reshape([N,N-1])
    u_x[:,0] = 1-u_x[:,1]
    u_x[:,-1] = -u_x[:,-2]
    u_y[0,:] = -u_y[1,:]
    u_y[-1,:] = -u_y[-2,:]

    return u_x, u_y

#
#          u_x[0,0]           u_x[1,0]           u_x[2,0]
#
#  u_y[0,0]   ----- u_y[1,0] --------- u_y[2,0] ---------
#             |                  |                  |
#             |                  |                  |
#          u_x[0,1]  p[0,0]   u_x[1,1]  p[1,0]   u_x[2,1]
#             |             u_x_star[0,0]      u_x_star[1,0]
#             |                  |                  |
#  u_y[0,1]   |---- u_y[1,1] ----+---- u_y[2,1] ----+----
#             |                  |                  |
#             |                  |                  |
#          u_x[0,2]  p[0,1]   u_x[1,2]  p[1,1]   u_x[2,2]
#             |             u_x_star[0,1]      u_x_star[1,1]
#
#                     ...                ...
#
#             |                  |                  |
#          u_x[0,N] p[0,N-1]  u_x[1,N] p[1,N-1]  u_x[2,N]
#             |             u_x_star[0,N-1]         |
#             |                  |                  |
#  u_y[0,N]   ----- u_y[1,N] --------- u_y[2,N] ---------
#                                                        
#         u_x[0,N+1]         u_x[1,N+1]         u_x[2,N+1]

def expand_u_x(u_x):
    u_x_expanded = zeros([u_x.shape[0] + 2, u_x.shape[1] + 2])
    u_x_expanded[1:-1,1:-1] = u_x
    u_x_expanded[1:-1,-1] =   -u_x[:,-1]  # south
    u_x_expanded[1:-1,0] = 2 - u_x[:,0]   # north, driven by lid
    return u_x_expanded

def calculate_A_for_u_x(u_x, u_y):
    # interpolation for CV face velocities
    u_x_e = (u_x[1:-1,1:-1] + u_x[2:,  1:-1]) / 2
    u_x_w = (u_x[:-2, 1:-1] + u_x[1:-1,1:-1]) / 2
    u_y_s = (u_y[2:-1,1:]  + u_y[1:-2,1:]) / 2
    u_y_n = (u_y[2:-1,:-1] + u_y[1:-2,:-1]) / 2
    nu = 1 / Re
    # coefficients
    A_e = u_x_e * dy  - nu * dy / dx
    A_w = -u_x_w * dy - nu * dy / dx
    A_n = u_y_n * dx  - nu * dx / dy
    A_s = -u_y_s * dx - nu * dx / dy
    A_p = -(A_e + A_w + A_n + A_s)
    return value(A_e), value(A_w), value(A_n), value(A_s), value(A_p)

def convection_diffusion_x(u_x_star, A_e, A_w, A_n, A_s, A_p, rhs_u_x):
    u_x_star_expanded = expand_u_x(u_x_star.reshape(A_p.shape))
    residual = A_p * u_x_star.reshape(A_p.shape) \
                   + A_e * u_x_star_expanded[2:,1:-1] \
                   + A_w * u_x_star_expanded[:-2,1:-1] \
                   + A_n * u_x_star_expanded[1:-1,:-2] \
                   + A_s * u_x_star_expanded[1:-1,2:] - rhs_u_x
    return ravel(residual)

def calculate_rhs_u_x(p):
    return ( - p[1:,:] + p[:-1,:]) * dy








#
#          u_x[0,0]           u_x[1,0]           u_x[2,0]
#
#  u_y[0,0]   ----- u_y[1,0] --------- u_y[2,0] ---------
#             |                  |                  |
#             |                  |                  |
#          u_x[0,1]  p[0,0]   u_x[1,1]  p[1,0]   u_x[2,1]
#             |                  |                  |       
#             |                  |                  |
#  u_y[0,1]   |---- u_y[1,1] ----+---- u_y[2,1] ----+----
#             |   u_y_star[0,0]  |   u_y_star[1,0]  |
#             |                  |                  |
#          u_x[0,2]  p[0,1]   u_x[1,2]  p[1,1]   u_x[2,2]
#             |                  |                  |       
#
#                     ...                ...
#             |  u_y_star[0,N-2] |  u_y_star[1,N-2] |
#
#             |                  |                  |
#          u_x[0,N] p[0,N-1]  u_x[1,N] p[1,N-1]  u_x[2,N]
#             |                  |                  |
#             |                  |                  |
#  u_y[0,N]   ----- u_y[1,N] --------- u_y[2,N] ---------
#                                                        
#         u_x[0,N+1]         u_x[1,N+1]         u_x[2,N+1]
def expand_u_y(u_y):
    u_y_expanded = zeros([u_y.shape[0] + 2,u_y.shape[1] + 2])
    u_y_expanded[1:-1,1:-1] = u_y
    u_y_expanded[0,1:-1] =  -u_y[1,:] # west
    u_y_expanded[-1,1:-1] = -u_y[-1,:] # east
    return u_y_expanded

def calculate_A_for_u_y(u_x, u_y):
    # interpolation for CV face velocities
    u_x_e = (u_x[1:,2:-1] + u_x[1:,1:-2]) / 2
    u_x_w = (u_x[0:-1,2:-1] + u_x[0:-1,1:-2]) / 2
    u_y_s = (u_y[1:-1,2:]  + u_y[1:-1,1:-1]) / 2
    u_y_n = (u_y[1:-1,1:-1] + u_y[1:-1,0:-2]) / 2
    nu = 1 / Re
    # coefficients
    A_e = u_x_e * dy  - nu * dy / dx
    A_w = -u_x_w * dy - nu * dy / dx
    A_n = u_y_n * dx  - nu * dx / dy
    A_s = -u_y_s * dx - nu * dx / dy
    A_p = -(A_e + A_w + A_n + A_s)
    return value(A_e), value(A_w), value(A_n), value(A_s), value(A_p)

def convection_diffusion_y(u_y_star, A_e, A_w, A_n, A_s, A_p, rhs_u_y):
    u_y_star_expanded = expand_u_y(u_y_star.reshape(A_p.shape))
    residual = A_p * u_y_star.reshape(A_p.shape)\
                   + A_e * u_y_star_expanded[2:,1:-1] \
                   + A_w * u_y_star_expanded[:-2,1:-1] \
                   + A_n * u_y_star_expanded[1:-1,:-2] \
                   + A_s * u_y_star_expanded[1:-1,2:] - rhs_u_y
    return ravel(residual)

def calculate_rhs_u_y(p):
    return (- p[:,1:] + p[:,:-1]) * dx


def calculate_A_for_p(A_for_u_x, A_for_u_y):
   
    a_e = ones([N,N]) 
    a_w = ones([N,N])
    a_n = ones([N,N])
    a_s = ones([N,N])
 
    a_e[:-1,:] = A_for_u_x[4]
    a_w[1:,:] = A_for_u_x[4]
    a_n[:,1:] = A_for_u_y[4]
    a_s[:,:-1] = A_for_u_y[4]    

    A_e = dy**2 / a_e
    A_w = dy**2 / a_w
    A_n = dx**2 / a_n
    A_s = dx**2 / a_s
    A_p = -(A_e + A_w + A_n + A_s)

    return value(A_e), value(A_w), value(A_n), value(A_s), value(A_p)


def calculate_rhs_p(u_x_star, u_y_star):
    u_x_star_full = zeros([N+1,N])
    u_x_star_full[1:-1,:] = u_x_star.reshape([N-1,N])
    u_y_star_full = zeros([N,N+1])
    u_y_star_full[:,1:-1] = u_y_star.reshape([N,N-1])
    
    u_w_star = u_x_star_full[:-1,:]
    u_e_star = u_x_star_full[1:,:]
    u_s_star = u_y_star_full[:,1:]
    u_n_star = u_y_star_full[:,:-1]
    # rhs = (rho*u_w - rho*u_e)*dy + (rho*u_s - rho*u_n)*dx
    
    return (u_w_star - u_e_star)*dy + (u_s_star - u_n_star)*dx
    

def pressure_correction(p_prime, A_e, A_w, A_n, A_s, A_p, rhs_p):
    p_prime_extend = zeros([N+2,N+2])
    p_prime_extend[1:-1,1:-1] = p_prime.reshape([N,N])
    residual = A_p * p_prime_extend[1:-1,1:-1] \
                   + A_e * p_prime_extend[2:,1:-1] \
                   + A_w * p_prime_extend[:-2,1:-1] \
                   + A_n * p_prime_extend[1:-1,:-2] \
                   + A_s * p_prime_extend[1:-1,2:] - rhs_p
    return ravel(residual)

def calculate_u_prime(p_prime, A_for_u_x, A_for_u_y):
    # u'e = -Se / Ap(u) * (PE' - PP')
    p_prime = p_prime.reshape([N, N])
    dp_prime_x = p_prime[1:,:] - p_prime[:-1,:]
    dp_prime_y = p_prime[:,1:] - p_prime[:,:-1]
    u_x_prime = -dp_prime_x * dy / A_for_u_x[4]
    u_y_prime = -dp_prime_y * dx / A_for_u_y[4]
    return u_x_prime.reshape([N-1, N]), u_y_prime.reshape([N, N-1])

# ---------------------- time integration --------------------- #
N = 50
dx = dy = 0.1
Re = 100

urf_p = 0.5
urf_u = 0.5

# initial conditions
u_x_interior = zeros([N-1, N])
u_y_interior = zeros([N, N-1])
p = zeros([N, N])

from matplotlib.pyplot import *

for iteration in range(5):
    u_x = expand_u_x(u_x_interior)
    u_y = expand_u_y(u_y_interior)
    
    # solve for u*
    rhs_u_x = calculate_rhs_u_x(p)
    A_for_u_x = calculate_A_for_u_x(u_x, u_y)
    u_x_star = solve(convection_diffusion_x, ravel(u_x_interior), args = A_for_u_x + (rhs_u_x,))

    # optional: replace u_x by expanded version of u_x_star

    # solve for v*
    rhs_u_y = calculate_rhs_u_y(p)
    A_for_u_y = calculate_A_for_u_y(u_x, u_y)
    u_y_star = solve(convection_diffusion_y, ravel(u_y_interior), args = A_for_u_y + (rhs_u_y,))
    
    # solve for p'
    rhs_p = calculate_rhs_p(u_x_star, u_y_star)
    A_for_p = calculate_A_for_p(A_for_u_x, A_for_u_y)
    p_prime = solve(pressure_correction, ravel(p), args = A_for_p + (rhs_p,))
    
    # update p with under-relaxation 
    p = p + urf_p * p_prime.reshape([N,N])
  
    # calculate u' from p'
    u_x_prime, u_y_prime = calculate_u_prime(p_prime, A_for_u_x, A_for_u_y)
    
    # calculate new u and v with no relaxation (intermediate values)
    # u_x_nr = u_x_star.reshape([N-1,N]) + u_x_prime
    # u_y_nr = u_y_star.reshape([N,N-1]) + u_y_prime
    
    # apply under-relaxation to u and v
    # u_x_interior = urf_u * u_x_nr + (1 - urf_u) * u_x_interior
    # u_y_interior = urf_u * u_y_nr + (1 - urf_u) * u_y_interior
    u_x_interior = u_x_star.reshape([N-1,N]) + urf_u * u_x_prime
    u_y_interior = u_y_star.reshape([N,N-1]) + urf_u * u_y_prime
    
    # compute cell center velocities and magnitude
    u_y_c = (u_y_interior[1:,:] + u_y_interior[:-1,:]) / 2.
    u_x_c = (u_x_interior[:,1:] + u_x_interior[:,:-1]) / 2.
    u_mag_interior = u_x_c**2 + u_y_c**2

    if iteration % 1 == 0:
        contourf(np.sqrt(u_mag_interior._value))
        show()
