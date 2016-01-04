import numpy as np
import matplotlib.pyplot as plt

# Adapted from C++ code by Dan Sunday:
#  Copyright 2000 softSurfer, 2012 Dan Sunday
#  This code may be freely used and modified for any purpose
#  providing that this copyright notice is included with it.
#  SoftSurfer makes no warranty for this code, and cannot be held
#  liable for any real or imagined damage resulting from its use.
#  Users of this code must verify correctness for their application.

#  isLeft(): tests if a point is Left|On|Right of an infinite line.
#     Input:  three points P0, P1, and P2
#     Return: >0 for P2 left of the line through P0 and P1
#             =0 for P2  on the line
#             <0 for P2  right of the line
#     See: Algorithm 1 "Area of Triangles and Polygons"

def isLeft(P0, P1, P2 ):
    return ( (P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] -  P0[0]) * (P1[1] - P0[1]) );
# ===================================================================


#  cn_PnPoly(): crossing number test for a point in a polygon
#       Input:   P = a point,
#                V[] = vertex points of a polygon V[n+1] with V[n]=V[0]
#       Return:  0 = outside, 1 = inside
#  This code is patterned after [Franklin, 2000]
def cn_PnPoly( P, V ):
    cn = 0;    #  the  crossing number counter

    #  loop through all edges of the polygon
    for i in range(V.shape[0]-1):    #  edge from V[i]  to V[i+1]
        if (((V[i,1] <= P[1]) and (V[i+1,1] > P[1])) or ((V[i,1] > P[1]) and (V[i+1,1] <=  P[1]))): #  a downward crossing
            #  compute  the actual edge-ray intersect x-coordinate
            vt = (P[1]  - V[i,1]) / (V[i+1,1] - V[i,1])
            if (P[0] <  V[i,0] + vt * (V[i+1,0] - V[i,0])): #  P.x < intersect
                 cn+=1;   #  a valid crossing of y=P.y right of P.x

    return bool(cn % 2);    #  0 if even (out), and 1 if  odd (in)

# ===================================================================


#  wn_PnPoly(): winding number test for a point in a polygon
#       Input:   P = a point,
#                V[] = vertex points of a polygon V[n+1] with V[n]=V[0]
#       Return:  wn = the winding number (=0 only when P is outside)
def wn_PnPoly( P, V ):

    wn = 0;    #  the  winding number counter

    #  loop through all edges of the polygon
    for i in range(V.shape[0]-1):    #  edge from V[i] to  V[i+1]
        if (V[i,1] <= P[1]):         #  start y <= P.y
            if (V[i+1,1]  > P[1]):      #  an upward crossing
                 if (isLeft( V[i], V[i+1], P) > 0):  #  P left of  edge
                     wn+=1           #  have  a valid up intersect

        else:                      #  start y > P.y (no test needed)
            if (V[i+1,1]  <= P[1]):     #  a downward crossing
                 if (isLeft( V[i], V[i+1], P) < 0):  #  P right of  edge
                     wn-=1           #  have  a valid down intersect

    return bool(wn)
# ===========

def plot_poly_test(V,n):
    fig1, ax1 = plt.subplots(1, 1)
    PX = np.random.uniform(V[:,0].min()-1, V[:,0].max()+1, (n,1))
    PY = np.random.uniform(V[:,1].min()-1, V[:,1].max()+1, (n,1))
    ax1.plot(V[:,0], V[:,1], 'b-')
    for i in range(n):
        if wn_PnPoly([PX[i],PY[i]], V ):
            ax1.plot(PX[i],PY[i], 'g.')
        else:
            ax1.plot(PX[i],PY[i], 'r.')