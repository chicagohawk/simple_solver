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

def showgrid(Xg, Yg):
    [plt.plot(x,y,'black') for x,y in zip(Xg,Yg)]
    [plt.plot(x,y,'black') for x,y in zip(Xg.transpose(),Yg.transpose())]
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    rin  = 1
    rout = 2
    theta = linspace(pi/2, -pi/2, 40)
    
    # interior boundary
    xin = cos(theta) * rin
    yin = sin(theta) * rin
    xin = hstack( [linspace(-.5, -.1, 5), xin, linspace(-.1, -1., 10), -1.-logspace(.1, 1., 5)/10.] )
    yin = hstack( [ones(5), yin, -ones(10), -ones(5)] )
    
    # exterior boundary
    xout = cos(theta) * rout
    yout = sin(theta) * rout
    xout = hstack( [linspace(-.5, -.1, 5), xout, linspace(-.1, -1., 10), -1.-logspace(.1, 1., 5)/10.] )
    yout = hstack( [2*ones(5), yout, -2*ones(10), -2*ones(5)] )
    
    # layer parameterization, # of layers == layer.size-1
    ratio = logspace(.1,.5,10)
    layer_in = (ratio - ratio.min()) / (ratio.max() - ratio.min()) * .5
    layer_out = -layer_in[:-1][::-1] + 1.
    layer = hstack([layer_in, layer_out])[::-1]
    grid = array( ones(layer.size-1) ,dtype=int)

    """
    layer = [logspace(1,.5,10), logspace(.1,.5,10)[::-1]
    layer = array([1., 0.9, 0.7, 0.3, 0.1, 0.])    # layer must start with 1 and end with 0
    grid = array([5, 3, 3, 3, 5], dtype=int)    # discretization for each layer
    """

    Xg, Yg = genMesh(xin, yin, xout, yout, layer, grid)
