from numpy import *
from matplotlib.pyplot import *
from pdb import set_trace
import re, string

rin  = 1
rout = 2

theta = linspace(pi/2, -pi/2, 20)

# interior boundary
xin = cos(theta) * rin
yin = sin(theta) * rin

# exterior boundary
xout = cos(theta) * rout
yout = sin(theta) * rout

# layer parameterization, # of layers == layer.size-1
layer = array([1., 0.8, 0.2, 0.])    # layer must start with 1 and end with 0
grid = array([10, 5, 10], dtype=int)    # discretization for each layer

f = open('blockMeshDict_template','r')
fstr = string.join( f.readlines() )
f.close()

X = []
Y = []

for ii in layer:
    X.append( (xout-xin)*ii + xin )
    Y.append( (yout-yin)*ii + yin )

X = vstack(X).transpose()
Y = vstack(Y).transpose()
Z = zeros(X.shape)

# Z-direction extrusion
# Each row of X,Y,Z corresponds to a pizza slice
# Each slice has 2*layer.size vertices
X = hstack([X, X])
Y = hstack([Y, Y])
Z = hstack([Z, ones(Z.shape)])

XYZ = vstack([X.ravel(), Y.ravel(), Z.ravel()])

# write vertices
vertex_str = ''
for ii in range(XYZ.shape[1]):
    vertex_str += '('+array_str( XYZ[:,ii] )[1:-1]+')\n\t'
pattern = '_VERTEX_'
fstr = re.sub(pattern, vertex_str, fstr)

# constants for a given geometry
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


# edges: use default straight lines

# boundary
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

f = open('blockMeshDict','w')
f.write(fstr)
f.close()

















