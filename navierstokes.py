import time
import sys
sys.path.append('../..')
import numpy as np
from numpad import *
import re, string, os
from pdb import set_trace
from matplotlib import pyplot as plt
import scipy.optimize as opt

Ni, Nj = 50, 20
def degrade(_adarray_):
    if isinstance(_adarray_, adarray):
        return _adarray_._value
    return _adarray_

def upgrade(_ndarray_):
    if isinstance(_ndarray_, np.ndarray):
        return array(_ndarray_)
    return _ndarray_

def genMesh(xin, yin, xout, yout, layer, grid):
    """
    generate 2D Ubend blockMesh for openFoam using inner and
    outer wall vertices coordinates.
	inputs ----
    	xin, yin:	inner wall coordinates	array
        xout, yout:	outer wall coordinates, array
        layer: 		radial partition, 	array
        grid:		each partition's grid, 	int array
    	outputs ----
        Xg, Yg: mesh coordinates
        write file blockMeshDict
    """
    # ======== INPUT CHECK ========
    assert(layer.size == grid.size+1)
    assert(layer[0]==1. and layer[-1]==0.)
    assert(all( np.diff(layer)<0 ))
    assert(all(grid>=1))
    assert(os.path.isfile('blockMeshDict_template'))

    f = open('blockMeshDict_template','r')
    # fstr = string.join( f.readlines() )
    fstr = ''.join(f.readlines())
    f.close()
    
    # ======== COORDINATE ========
    X, Y = [], []			# block mesh
    Xg, Yg = [], []			# grid mesh
    
    for ii in layer:
        X.append( (xout-xin)*ii + xin )
        Y.append( (yout-yin)*ii + yin )

    X = np.vstack(X).transpose()
    Y = np.vstack(Y).transpose()
    Z = np.zeros(X.shape)    

    spacing = - np.diff(layer) / grid	# grid's radial spacing (normalized)
    spacing = np.hstack( [ np.repeat(spacing[ii], grid[ii]) for ii in range(spacing.size) ] )
    # set_trace()
    spacing = layer.copy()
    for ii in spacing:
        Xg.append( (xout-xin)*ii + xin )
        Yg.append( (yout-yin)*ii + yin )
   
    Xg = np.vstack(Xg).transpose()
    Yg = np.vstack(Yg).transpose()

    # Z-direction extrusion
    # Each row of X,Y,Z corresponds to a pizza slice
    # Each slice has 2*layer.size vertices
    X = np.hstack([X, X])
    Y = np.hstack([Y, Y])
    Z = np.hstack([Z, 0.1*np.ones(Z.shape)])
    
    XYZ = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # ======== VERTEX ========
    vertex_str = ''
    for ii in range(XYZ.shape[1]):
        vertex_str += '('+np.array_str( XYZ[:,ii] )[1:-1]+')\n\t'
    pattern = '_VERTEX_'
    fstr = re.sub(pattern, vertex_str, fstr)

    # ======== BLOCKS ========
    endv = (xin.size-1)*2*layer.size    # first vertex of last slice
    # blocks by slice
    # simpleGrading depends on jj
    Hex = []
    Grade = []
    for ii in np.r_[0 : endv : 2*layer.size]:
        for jj in range(layer.size-1):
            hexij = np.array( [ii, ii+1, ii+1+layer.size*2, ii+layer.size*2, \
                               ii+layer.size, ii+1+layer.size, \
                               ii+layer.size*3+1, ii+layer.size*3] , dtype=int)
            hexij += jj
            Hex.append(hexij)
            Grade.append(jj)    # mesh refinement
    
    Hex_str = ''
    for ii in range(len(Hex)):
        Hex_str += 'hex ' + '(' + np.array_str(Hex[ii])[1:-1] + ')\n\t'
        Hex_str += '(' + str(grid[Grade[ii]]) + ' 1 1)\n\t'
        Hex_str += 'simpleGrading (1 1 1)\n\n\t'
    pattern = '_HEX_'
    fstr = re.sub(pattern, Hex_str, fstr)
    
    
    # ======== EDGE ========
    pass
    
    # ======== BOUNDARY ========
    # inlet
    inlet_str = ''
    for ii in range(layer.size-1):
        faces = np.array([ii, ii+1, ii+1+layer.size, ii+layer.size],dtype=int)
        inlet_str += '('+np.array_str(faces)[1:-1]+')\n\t\t'
    pattern = '_INLET_FACES_'
    fstr = re.sub(pattern, inlet_str, fstr)
    
    # outlet
    outlet_str = ''
    for ii in range(layer.size-1):
        faces = np.array([ii+1, ii, ii+layer.size, ii+layer.size+1],dtype=int)+endv
        outlet_str += '('+np.array_str(faces)[1:-1]+')\n\t\t'
    pattern = '_OUTLET_FACES_'
    fstr = re.sub(pattern, outlet_str, fstr)
    
    # wall_inner
    wallin_str = ''
    for ii in range(xin.size-1):
        faces = np.array( [layer.size*(2*ii+1) -1, layer.size*(2*ii+3) -1,
                           layer.size*(2*ii+4) -1, layer.size*(2*ii+2) -1], dtype=int)
        wallin_str += '('+np.array_str(faces)[1:-1]+')\n\t\t'
    pattern = '_WALL_IN_FACES_'
    fstr = re.sub(pattern, wallin_str, fstr)
    
    # wall_outer
    wallout_str = ''
    for ii in range(xin.size-1):
        faces = np.array( [layer.size*(2*ii), layer.size*(2*ii+1),
                           layer.size*(2*ii+3), layer.size*(2*ii+2)], dtype=int)
        wallout_str += '('+np.array_str(faces)[1:-1]+')\n\t\t'
    pattern = '_WALL_OUT_FACES_'
    fstr = re.sub(pattern, wallout_str, fstr)
    
    # ======== WRITE TO FILE ========
    f = open('blockMeshDict','w')
    f.write(fstr)
    f.close()

    return Xg, Yg


def showgrid(Xg, Yg):
    [plt.plot(x,y,'black') for x,y in zip(Xg,Yg)]
    [plt.plot(x,y,'black') for x,y in zip(Xg.transpose(),Yg.transpose())]
    plt.axis('equal')
    plt.show()


def MikeMesh():
    # grid count in streamwise direction
    n_1 = 35   # 1st straight section
    n_2 = 35   # 2nd straight section
    n_b = 30   # bend section
    n_s = 4    # sponge region
    
    # straight section lengths
    L_1 = 2.5
    L_2 = 2.5
    L_s = 1.

    # straight section streamwise grid size
    dx1 = L_1/n_1
    dx2 = L_2/n_2

    # duct width, bend inner and outer radius, grid angle
    Width = 1.
    rin  = .25
    rout = Width + rin
    theta = np.linspace(np.pi/2, -np.pi/2, n_b+1)

    # turn off or on sponge at exit, 1 = on, 0 = off
    sponge = 1
    
    # bend region, interior
    xin = np.cos(theta) * rin
    yin = np.sin(theta) * rin
    # boundary bend, exterior
    xout = np.cos(theta) * rout
    yout = np.sin(theta) * rout

    # sponge region
    Rsponge = 4.
    cratio_sponge = pow(Rsponge, 1./(n_s-1.))
    dstart_sponge = L_s*(1-cratio_sponge)/(1-pow(cratio_sponge,n_s))
    xin_test = [dstart_sponge]
    xin_test2 = [dstart_sponge]
    for i in range(0,n_s-1):
       xin_new = xin_test[i]*cratio_sponge
       xin_test = np.hstack( [xin_test, xin_new] )
       xin_test2 = np.hstack( [xin_test2, sum(xin_test)] )
    xin_test2 = -L_2 - xin_test2
    
    xin_sponge = xin_test2
    xout_sponge = xin_sponge
    yin_sponge = -rin*np.ones(n_s)
    yout_sponge = -rout*np.ones(n_s)

    # complete boundary
    if sponge == 1:
       # interior boundary
       xin = np.hstack( [np.linspace(-L_1, -dx1, n_1), xin, 
                         np.linspace(-dx2, -L_2, n_2), xin_sponge] )
       yin = np.hstack( [rin*np.ones(n_1), yin, -rin*np.ones(n_2), yin_sponge] )
       # exterior boundary
       xout = np.hstack( [np.linspace(-L_1, -dx1, n_1), xout, 
                          np.linspace(-dx2, -L_2, n_2), xout_sponge] )
       yout = np.hstack( [rout*np.ones(n_1), yout, -rout*np.ones(n_2), yout_sponge] )
    else:
       n_s = 0
       # interior boundary
       xin = np.hstack( [np.linspace(-L_1, -dx1, n_1), xin, 
                         np.linspace(-dx2, -L_2, n_2)] )
       yin = np.hstack( [rin*np.ones(n_1), yin, -rin*np.ones(n_2)] )
       # exterior boundary
       xout = np.hstack( [np.linspace(-L_1, -dx1, n_1), xout, np.linspace(-dx2, -L_2, n_2)] )
       yout = np.hstack( [rout*np.ones(n_1), yout, -rout*np.ones(n_2)] )
    
    # spanwise layers
    # n = full width number of cells (must be even for symmetric grading)
    # Rspan = overall grading ratio (max cell size to min cell size)
    # cratio = cell expansion ratio (adjacent cell size ratio)
    nspan = 40
    n_halfspan = nspan/2		
    Rspan = 150.
    cratio = pow(Rspan, 1./(n_halfspan-1.))
    dstart = Width/2 * (1-cratio)/(1-pow(cratio,n_halfspan))
    span_test = [dstart]
    span_test2 = [dstart]

    for i in range(0,int(n_halfspan)-1):
       span_new = span_test[i]*cratio
       span_test = np.hstack( [span_test, span_new] )
       span_test2 = np.hstack( [span_test2, sum(span_test)] )
    span_test2 = np.hstack( [0, span_test2] )
    layer_in = span_test2

    n_cells = nspan*(n_1 + n_2 + n_b + n_s)
    
    # layer parameterization, # of layers == layer.size-1
    ratio = np.logspace(.1,.5,10)
    #layer_in = (ratio - ratio.min()) / (ratio.max() - ratio.min()) * .5
    layer_out = -layer_in[:-1][::-1] + 1.
    layer = np.hstack([layer_in, layer_out])[::-1]
    grid = np.array( np.ones(layer.size-1) ,dtype=int)

    Xg, Yg = genMesh(xin, yin, xout, yout, layer, grid)
    return Xg, Yg


def extend(w_interior, geo):
    '''
    Extend the conservative variables into ghost cells using boundary condition
    '''
    w = zeros([4, Ni+2, Nj+2])
    w[:,1:-1,1:-1] = w_interior.reshape([4, Ni, Nj])

    # inlet
    rho, u, v, E, p = primative(w[:,1,1:-1], coef)
    c2 = 1.4 * p / rho
    c = sqrt(c2)
    mach2 = u**2 / c2
    rhot = rho * (1 + 0.2 * mach2)**2.5
    pt = p * (1 + 0.2 * mach2)**3.5

    d_rho = 1 - rho
    d_pt = pt_in - pt
    d_u = d_pt / (rho * (u + c))
    d_p = rho * c * d_u

    rho = rho + d_rho
    u = u + d_u
    p = p + d_p
    w[0,0,1:-1] = rho
    w[1,0,1:-1] = rho * u
    w[2,0,1:-1] = 0
    w[3,0,1:-1] = p / 0.4 + 0.5 * rho * u**2

    # outlet
    w[:,-1,1:-1] = w[:,-2,1:-1]
    rho, u, v, E, p = primative(w[:,-1,1:-1], coef)
    p = p_out
    w[3,-1,1:-1] = p / (1.4 - 1) + 0.5 * rho * (u**2 + v**2)

    # walls
    w[:,:,0] = w[:,:,1]
    rhoU_n = sum(w[1:3,1:-1,0] * geo.normal_j[:,:,0], 0)
    w[1:3,1:-1,0] -= 2 * rhoU_n * geo.normal_j[:,:,0]

    w[:,:,-1] = w[:,:,-2]
    rhoU_n = sum(w[1:3,1:-1,-1] * geo.normal_j[:,:,-1], 0)
    w[1:3,1:-1,-1] -= 2 * rhoU_n * geo.normal_j[:,:,-1]

    return w
    
def primative(w , coef):
    '''
    Transform conservative variables into primative ones
    '''
    rho = w[0]
    u = w[1] / rho
    v = w[2] / rho
    E = w[3]
    p = coef[0] * E + coef[1] * (u*w[1] + v*w[2])
    # 0.4 * (E - 0.5 * (u * w[1] + v * w[2])) * coef
    return rho, u, v, E, p

def grad_dual(phi, geo):
    '''
    Gradient on the nodes assuming zero boundary conditions
    '''
    dxy_i = 0.5 * (geo.dxy_i[:,1:,:] + geo.dxy_i[:,:-1,:])
    dxy_j = 0.5 * (geo.dxy_j[:,:,1:] + geo.dxy_j[:,:,:-1])
    phi_i = array([phi * dxy_i[1], -phi * dxy_i[0]])
    phi_j = array([-phi * dxy_j[1], phi * dxy_j[0]])

    grad_phi = zeros(geo.xy.shape)
    grad_phi[:,:-1,:-1] += phi_i + phi_j
    grad_phi[:,1:,:-1] += -phi_i + phi_j
    grad_phi[:,:-1,1:] += phi_i - phi_j
    grad_phi[:,1:,1:] += -phi_i - phi_j

    area = zeros(geo.xy.shape[1:])
    area[:-1,:-1] += geo.area
    area[1:,:-1] += geo.area
    area[:-1,1:] += geo.area
    area[1:,1:] += geo.area

    return 2 * grad_phi / area

def ns_flux(rho, u, v, E, p, grad_u):
    # viscous stress
    dudx, dudy, dvdx, dvdy = grad_u
    mu_t = 0 # blabla
    sigma_xx = (mu + mu_t) * (2 * dudx - 2./3 * (dudx + dvdy))
    sigma_yy = (mu + mu_t) * (2 * dvdy - 2./3 * (dudx + dvdy))
    sigma_xy = (mu + mu_t) * (dudy + dvdx)

    F = array([rho * u, rho * u**2 + p - sigma_xx,
                        rho * u * v    - sigma_xy,
                        u * (E + p)    - sigma_xx * u - sigma_xy * v])
    G = array([rho * v, rho * u * v    - sigma_xy,
                        rho * v**2 + p - sigma_yy,
                        v * (E + p)    - sigma_xy * u - sigma_yy * v])
    return F, G

def sponge_flux(c_ext, w_ext, geo):
    ci = 0.5 * (c_ext[1:,1:-1] + c_ext[:-1,1:-1])
    cj = 0.5 * (c_ext[1:-1,1:] + c_ext[1:-1,:-1])

    a = geo.area
    ai = vstack([a[:1,:], (a[1:,:] + a[:-1,:]) / 2, a[-1:,:]])
    aj = hstack([a[:,:1], (a[:,1:] + a[:,:-1]) / 2, a[:,-1:]])

    wxx = (w_ext[:,2:,1:-1] + w_ext[:,:-2,1:-1] - 2 * w_ext[:,1:-1,1:-1]) / 3.
    wyy = (w_ext[:,1:-1,2:] + w_ext[:,1:-1,:-2] - 2 * w_ext[:,1:-1,1:-1]) / 3.
    # second order dissipation at boundary, fourth order in the interior
    Fi = -0.5 * ci * ai * (w_ext[:,1:,1:-1] - w_ext[:,:-1,1:-1])
    Fi[:,1:-1,:] = 0.5 * (ci * ai)[1:-1,:] * (wxx[:,1:,:] - wxx[:,:-1,:])
    Fj = -0.5 * cj * aj * (w_ext[:,1:-1,1:] - w_ext[:,1:-1,:-1])
    Fj[:,:,1:-1] = 0.5 * (cj * aj)[:,1:-1] * (wyy[:,:,1:] - wyy[:,:,:-1])
    return Fi, Fj

def ns_kec(w, w0, geo, dt, coef):
    '''
    Kinetic energy conserving scheme with no numerical viscosity
    '''
    w_ext = extend(w, geo)
    rho, u, v, E, p = primative(w_ext, coef)
    c = sqrt(1.4 * p / rho)
    # velocity gradient on nodes
    dudx, dudy = grad_dual(u[1:-1,1:-1], geo)
    dvdx, dvdy = grad_dual(v[1:-1,1:-1], geo)
    duv_dxy = array([dudx, dudy, dvdx, dvdy])
    # interface average
    rho_i = 0.5 * (rho[1:,1:-1] + rho[:-1,1:-1])
    rho_j = 0.5 * (rho[1:-1,1:] + rho[1:-1,:-1])
    u_i = 0.5 * (u[1:,1:-1] + u[:-1,1:-1])
    u_j = 0.5 * (u[1:-1,1:] + u[1:-1,:-1])
    v_i = 0.5 * (v[1:,1:-1] + v[:-1,1:-1])
    v_j = 0.5 * (v[1:-1,1:] + v[1:-1,:-1])
    E_i = 0.5 * (E[1:,1:-1] + E[:-1,1:-1])
    E_j = 0.5 * (E[1:-1,1:] + E[1:-1,:-1])
    p_i = 0.5 * (p[1:,1:-1] + p[:-1,1:-1])
    p_j = 0.5 * (p[1:-1,1:] + p[1:-1,:-1])
    # interface strain rate averged from dual mesh (nodal) values
    duv_dxy_i = 0.5 * (duv_dxy[:,:,1:] + duv_dxy[:,:,:-1])
    duv_dxy_j = 0.5 * (duv_dxy[:,1:,:] + duv_dxy[:,:-1,:])
    # inlet and outlet have no viscous stress
    duv_dxy_i[:,[0,-1],:] = 0
    # interface flux
    f_i, g_i = ns_flux(rho_i, u_i, v_i, E_i, p_i, duv_dxy_i)
    f_j, g_j = ns_flux(rho_j, u_j, v_j, E_j, p_j, duv_dxy_j)
    Fi = + f_i * geo.dxy_i[1] - g_i * geo.dxy_i[0]
    Fj = - f_j * geo.dxy_j[1] + g_j * geo.dxy_j[0]
    # sponge
    Fi_s, Fj_s = sponge_flux(c, w_ext, geo)
    Fi += 0.5 * Fi_s
    Fj += 0.5 * Fj_s
    # residual
    divF = (Fi[:,1:,:] - Fi[:,:-1,:] + Fj[:,:,1:] - Fj[:,:,:-1]) / geo.area
    return (w - w0) / dt + ravel(divF)


# -------------------------- geometry ------------------------- #
class geo2d:
    def __init__(self, xy):
        xy = array(xy)
        self.xy = xy
        self.xyc = (xy[:,1:,1:]  + xy[:,:-1,1:] + \
                    xy[:,1:,:-1] + xy[:,:-1,:-1]) / 4

        self.dxy_i = xy[:,:,1:] - xy[:,:,:-1]
        self.dxy_j = xy[:,1:,:] - xy[:,:-1,:]

        self.L_j = sqrt(self.dxy_j[0]**2 + self.dxy_j[1]**2)
        self.normal_j = array([self.dxy_j[1] / self.L_j,
                              -self.dxy_j[0] / self.L_j])

        self.area = self.tri_area(self.dxy_i[:,:-1,:], self.dxy_j[:,:,1:]) \
                  + self.tri_area(self.dxy_i[:,1:,:], self.dxy_j[:,:,:-1]) \

    def tri_area(self, xy0, xy1):
        return 0.5 * (xy0[1] * xy1[0] - xy0[0] * xy1[1])
        

# ----------------------- visualization --------------------------- #
def vis(w, geo, coef):
    '''
    Visualize Mach number, non-dimensionalized stagnation and static pressure
    '''
    def avg(a):
        return 0.25 * (a[1:,1:] + a[1:,:-1] + a[:-1,1:] + a[:-1,:-1])

    import numpy as np
    rho, u, v, E, p = primative(value(extend(w, geo)), coef)
    x, y = value(geo.xy)
    xc, yc = value(geo.xyc)
    
    c2 = 1.4 * p / rho
    M = degrade(sqrt((u**2 + v**2) / c2))
    pt = degrade(p * (1 + 0.2 * M**2)**3.5)

    plt.subplot(2,2,1)
    plt.contourf(x, y, avg(M), 100)
    plt.colorbar()
    plt.quiver(xc, yc, u[1:-1,1:-1], v[1:-1,1:-1])
    plt.axis('scaled')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Mach')
    plt.draw()
    
    plt.subplot(2,2,2)
    pt_frac = (pt - p_out) / (pt_in - p_out)
    plt.contourf(x, y, avg(pt_frac), 100)
    plt.colorbar()
    plt.axis('scaled')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('pt')
    plt.draw()
    
    plt.subplot(2,2,3)
    p_frac = degrade( (p - p_out) / (pt_in - p_out) )
    plt.contourf(x, y, avg(p_frac), 100)
    plt.colorbar()
    plt.axis('scaled')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('p')
    plt.draw()


# ---------------------- time integration --------------------- #
def var_grad(coef, lasso_reg):
    geometry = 'bend'
    
    if geometry == 'nozzle':
        # Ni, Nj = 100, 40
        x = linspace(-15,25,Ni+1)
        y = sin(linspace(-np.pi/2, np.pi/2, Nj+1))
        a = ones(Ni+1)
        a[np.abs(x) < 10] = 1 - (1 + cos(x[np.abs(x) < 10] / 10 * np.pi)) * 0.2
        
        y, x = np.meshgrid(y, x)
        y *= 5 * a[:,np.newaxis]
    
    elif geometry == 'bend':
        Ni, Nj = 40, 14
        # Ni, Nj = 200, 40
        theta = linspace(0, np.pi, Ni/2+1)
        r = 15 + 5 * sin(linspace(-np.pi/2, np.pi/2, Nj+1))
        r, theta = meshgrid(r, theta)
        x, y = r * sin(theta), r * cos(theta)
    
        dx = 15 * 2 * np.pi / Ni
        y0, y1 = y[0,:], y[-1,:]
        y0, x0 = meshgrid(y0, dx * arange(-Ni/4, 0))
        y1, x1 = meshgrid(y1, -dx * arange(1, 1 + Ni/4))
        
        x, y = vstack([x0, x, x1]), vstack([y0, y, y1])
    
    elif geometry == 'mike':
        x, y = MikeMesh()
        (Ni, Nj) = x.shape
        Ni -= 1
        Nj -= 1
    
    np.save('geo.npy', value(array([x, y])))
    geo = geo2d([x, y])
    
    t, dt = 0, 1./Nj
    
    pt_in = 1.2E5
    p_out = 1E5
    mu = 1
    
    w = zeros([4, Ni, Nj])
    w[0] = 1
    w[3] = 1E5 / (1.4 - 1)
    
    w0 = ravel(w)
    
    for i in range(100):
        print('i = ', i, 't = ', t)
        coef = upgrade(coef)
        (w, good_quality) = solve(ns_kec, w0, args=(w0, geo, dt, coef), rel_tol=1E-7, abs_tol=1E-6)
    
        if not good_quality:
            dt *= 0.125
            continue
    
        if w._n_Newton == 1:
            break
        elif w._n_Newton < 10:
            w0 = w
            dt *= 2
        elif w._n_Newton < 20:
            w0 = w
        else:
            dt *= 0.125
            continue
        t += dt
        w0.obliviate()
    
        #if i % 10 == 0:
        #    vis(w, geo, coef)
        #    plt.show(block=True)
    
    print('Final, t = inf')
    dt = np.inf
    w = solve(ns_kec, w0, args=(w0, geo, dt, coef), rel_tol=1E-6, abs_tol=1E-4)

    w_standard = np.loadtxt('standard_ubend.txt')
    
    err_r = linalg.norm(w[:1600] - w_standard[:1600],2) + lasso_reg*linalg.norm(coef,2)
    err_u = linalg.norm(w[1600:3200] - w_standard[1600:3200],2) + lasso_reg*linalg.norm(coef,2)
    err_v = linalg.norm(w[3200:4800] - w_standard[3200:4800],2 + lasso_reg* linalg.norm(coef,2))
    err_E = linalg.norm(w[4800:] - w_standard[4800:],2) + lasso_reg*linalg.norm(coef,2)
    
    grad_r = err_r.diff(coef)
    grad_u = err_u.diff(coef)
    grad_v = err_v.diff(coef)
    grad_E = err_E.diff(coef)
    set_trace()

    return (err_E, grad_E)

coef = array([1.,1.])
result = opt.minimize(var_grad, degrade(coef), args=(1e-5,), jac=True)
set_trace()
