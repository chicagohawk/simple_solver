from numpy import *
from matplotlib import pyplot as plt
from pdb import set_trace
import re, string, os


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
    assert(all( diff(layer)<0 ))
    assert(all(grid>=1))
    assert(os.path.isfile('blockMeshDict_template'))

    f = open('blockMeshDict_template','r')
    fstr = string.join( f.readlines() )
    f.close()
    
    # ======== COORDINATE ========
    X, Y = [], []			# block mesh
    Xg, Yg = [], []			# grid mesh
    
    for ii in layer:
        X.append( (xout-xin)*ii + xin )
        Y.append( (yout-yin)*ii + yin )

    X = vstack(X).transpose()
    Y = vstack(Y).transpose()
    Z = zeros(X.shape)    

    spacing = - diff(layer) / grid	# grid's radial spacing (normalized)
    spacing = hstack( [ repeat(spacing[ii], grid[ii]) for ii in range(spacing.size) ] )
    spacing = hstack([array([0]), cumsum(spacing)])[::-1]
    for ii in spacing:
        Xg.append( (xout-xin)*ii + xin )
        Yg.append( (yout-yin)*ii + yin )
   
    Xg = vstack(Xg).transpose()
    Yg = vstack(Yg).transpose()
   
    # Z-direction extrusion
    # Each row of X,Y,Z corresponds to a pizza slice
    # Each slice has 2*layer.size vertices
    X = hstack([X, X])
    Y = hstack([Y, Y])
    Z = hstack([Z, 0.1*ones(Z.shape)])
    
    XYZ = vstack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # ======== VERTEX ========
    vertex_str = ''
    for ii in range(XYZ.shape[1]):
        vertex_str += '('+array_str( XYZ[:,ii] )[1:-1]+')\n\t'
    pattern = '_VERTEX_'
    fstr = re.sub(pattern, vertex_str, fstr)
    
    # ======== BLOCKS ========
    endv = (xin.size-1)*2*layer.size    # first vertex of last slice
    # blocks by slice
    # simpleGrading depends on jj
    Hex = []
    Grade = []
    for ii in r_[0 : endv : 2*layer.size]:
        for jj in range(layer.size-1):
            hexij = array( [ii, ii+1, ii+1+layer.size*2, ii+layer.size*2, \
                            ii+layer.size, ii+1+layer.size, \
                            ii+layer.size*3+1, ii+layer.size*3] , dtype=int)
            hexij += jj
            Hex.append(hexij)
            Grade.append(jj)    # mesh refinement
    
    Hex_str = ''
    for ii in range(len(Hex)):
        Hex_str += 'hex ' + '(' + array_str(Hex[ii])[1:-1] + ')\n\t'
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
        faces = array([ii, ii+1, ii+1+layer.size, ii+layer.size],dtype=int)
        inlet_str += '('+array_str(faces)[1:-1]+')\n\t\t'
    pattern = '_INLET_FACES_'
    fstr = re.sub(pattern, inlet_str, fstr)
    
    # outlet
    outlet_str = ''
    for ii in range(layer.size-1):
        faces = array([ii+1, ii, ii+layer.size, ii+layer.size+1],dtype=int)+endv
        outlet_str += '('+array_str(faces)[1:-1]+')\n\t\t'
    pattern = '_OUTLET_FACES_'
    fstr = re.sub(pattern, outlet_str, fstr)
    
    # wall_inner
    wallin_str = ''
    for ii in range(xin.size-1):
        faces = array( [layer.size*(2*ii+1) -1, layer.size*(2*ii+3) -1,
                        layer.size*(2*ii+4) -1, layer.size*(2*ii+2) -1], dtype=int)
        wallin_str += '('+array_str(faces)[1:-1]+')\n\t\t'
    pattern = '_WALL_IN_FACES_'
    fstr = re.sub(pattern, wallin_str, fstr)
    
    # wall_outer
    wallout_str = ''
    for ii in range(xin.size-1):
        faces = array( [layer.size*(2*ii), layer.size*(2*ii+1),
                        layer.size*(2*ii+3), layer.size*(2*ii+2)], dtype=int)
        wallout_str += '('+array_str(faces)[1:-1]+')\n\t\t'
    pattern = '_WALL_OUT_FACES_'
    fstr = re.sub(pattern, wallout_str, fstr)
    
    # ======== WRITE TO FILE ========
    f = open('blockMeshDict','w')
    f.write(fstr)
    f.close()

    return Xg, Yg

def showgrid():
    [plt.plot(x,y,'black') for x,y in zip(Xg,Yg)]
    [plt.plot(x,y,'black') for x,y in zip(Xg.transpose(),Yg.transpose())]
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    
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
    theta = linspace(pi/2, -pi/2, n_b+1)

    # turn off or on sponge at exit, 1 = on, 0 = off
    sponge = 1
    
    # bend region, interior
    xin = cos(theta) * rin
    yin = sin(theta) * rin
    # boundary bend, exterior
    xout = cos(theta) * rout
    yout = sin(theta) * rout

    # sponge region
    Rsponge = 4.
    cratio_sponge = pow(Rsponge, 1./(n_s-1.))
    dstart_sponge = L_s*(1-cratio_sponge)/(1-pow(cratio_sponge,n_s))
    xin_test = [dstart_sponge]
    xin_test2 = [dstart_sponge]
    for i in range(0,n_s-1):
       xin_new = xin_test[i]*cratio_sponge
       xin_test = hstack( [xin_test, xin_new] )
       xin_test2 = hstack( [xin_test2, sum(xin_test)] )
    xin_test2 = -L_2 - xin_test2
    
    xin_sponge = xin_test2
    xout_sponge = xin_sponge
    yin_sponge = -rin*ones(n_s)
    yout_sponge = -rout*ones(n_s)

    # complete boundary
    if sponge == 1:
       # interior boundary
       xin = hstack( [linspace(-L_1, -dx1, n_1), xin, linspace(-dx2, -L_2, n_2), xin_sponge] )
       yin = hstack( [rin*ones(n_1), yin, -rin*ones(n_2), yin_sponge] )
       # exterior boundary
       xout = hstack( [linspace(-L_1, -dx1, n_1), xout, linspace(-dx2, -L_2, n_2), xout_sponge] )
       yout = hstack( [rout*ones(n_1), yout, -rout*ones(n_2), yout_sponge] )
    else:
       n_s = 0
       # interior boundary
       xin = hstack( [linspace(-L_1, -dx1, n_1), xin, linspace(-dx2, -L_2, n_2)] )
       yin = hstack( [rin*ones(n_1), yin, -rin*ones(n_2)] )
       # exterior boundary
       xout = hstack( [linspace(-L_1, -dx1, n_1), xout, linspace(-dx2, -L_2, n_2)] )
       yout = hstack( [rout*ones(n_1), yout, -rout*ones(n_2)] )
    
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
    for i in range(0,n_halfspan-1):
       span_new = span_test[i]*cratio
       span_test = hstack( [span_test, span_new] )
       span_test2 = hstack( [span_test2, sum(span_test)] )
    span_test2 = hstack( [0, span_test2] )
    layer_in = span_test2

    n_cells = nspan*(n_1 + n_2 + n_b + n_s)
    
    # layer parameterization, # of layers == layer.size-1
    ratio = logspace(.1,.5,10)
    #layer_in = (ratio - ratio.min()) / (ratio.max() - ratio.min()) * .5
    layer_out = -layer_in[:-1][::-1] + 1.
    layer = hstack([layer_in, layer_out])[::-1]
    grid = array( ones(layer.size-1) ,dtype=int)

    """
    layer = [logspace(1,.5,10), logspace(.1,.5,10)[::-1]
    layer = array([1., 0.9, 0.7, 0.3, 0.1, 0.])    # layer must start with 1 and end with 0
    grid = array([5, 3, 3, 3, 5], dtype=int)    # discretization for each layer
    """

    Xg, Yg = genMesh(xin, yin, xout, yout, layer, grid)
    set_trace()
