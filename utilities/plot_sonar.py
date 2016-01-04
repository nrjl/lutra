import pickle
import numpy as np
import GPy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import os.path
plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
#plt.rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Sans Serif']})
plt.rc('text', usetex=True)

REBUILD_MODEL = False

home_dir = os.path.expanduser("~")
fh = open(home_dir+'/catkin_ws/src/ros_lutra/data/sonar_data.pkl', 'rb')
poses = pickle.load(fh)
sonar = pickle.load(fh)
temps = pickle.load(fh)
fh.close()

dd = sonar[:,3] < 10.0
dd[0:375] = False
X = sonar[dd,1:3]
y = np.atleast_2d(sonar[dd,3]).transpose()
ymean = y.mean()
    
model_file = home_dir+'/catkin_ws/src/ros_lutra/data/IrelandLnModel.pkl'

if REBUILD_MODEL or not os.path.isfile(model_file):
    print "Building GP model..."
    GP_model = GPy.models.GPRegression(X,y-ymean,GPy.kern.RBF(2))
    GP_model.kern.lengthscale = 20
    GP_model.kern.variance = 10
    GP_model.Gaussian_noise.variance = 0.25
    print "Optimizing..."
    GP_model.optimize()
    print "Done."
    fh = open(model_file, 'wb')
    pickle.dump(GP_model, fh)
    pickle.dump(ymean, fh)
    fh.close()
else:
    fh = open(model_file, 'rb')
    GP_model = pickle.load(fh)
    ymean = pickle.load(fh)
    fh.close()

xres,yres = (30, 40)
Xt = np.linspace(min(X[:,0]), max(X[:,0]), xres)
Yt = np.linspace(min(X[:,1]), max(X[:,1]), yres)
Xtemp, Ytemp = np.meshgrid(Xt, Yt)
    
Xfull = np.vstack([Xtemp.ravel(order='C'), Ytemp.ravel(order='C')]).transpose()
Yfull, varYfull = GP_model.predict(Xfull)
Yfull+=ymean
Yfull = np.reshape(Yfull, (yres,xres), order='C')
varYfull = np.reshape(varYfull, (yres,xres), order='C')

fig = plt.figure(1)
fig.clf()
ax = fig.add_subplot(111, projection='3d')
hp = ax.scatter(sonar[dd,1], sonar[dd,2], -sonar[dd,3], c=-sonar[dd,3])
ax.set_xlabel('Easting (m)')
ax.set_ylabel('Northing (m)')
ax.set_zlabel('Depth (m)')
#ax.set_aspect('equal')
ax.set_zlim(bottom=-5, top=0)

hs = ax.plot_surface(Xtemp, Ytemp, -Yfull, rstride=1, cstride=1, 
    alpha=0.6, linewidth=0.2, # edgecolors='face')
    cmap=cm.jet)
    #facecolors=cm.jet(varYfull/varYfull.max()))
# Bounds
#hu = ax.plot_wireframe(Xtemp, Ytemp, -Yfull+np.sqrt(varYfull), rstride=1, cstride=1, 
#    linewidth=0.2)
#hl = ax.plot_wireframe(Xtemp, Ytemp, -Yfull-np.sqrt(varYfull), rstride=1, cstride=1, 
#    linewidth=0.2)

V_Ireland = np.array([[0,0], [-43,-38],[-70,-94], [-60,-150],[0,-180],[54,-152],[85,-70],[0,0]])

fig2, ax2 = plt.subplots(1, 2)
fig2.set_size_inches(12,6,forward=True)
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
levels = np.arange(-5, -1, 0.25, dtype=float)
ax2[0].contourf(Xtemp, Ytemp, -Yfull, levels)
cs = ax2[0].contour(Xtemp, Ytemp, -Yfull, levels, colors='k', linewidth=1)
plt.clabel(cs, inline=1, fontsize=10, colors='k')
ax2[0].set_title('Depth ($m$)')
ax2[1].contourf(Xtemp, Ytemp, varYfull, 6)
cs = ax2[1].contour(Xtemp, Ytemp, varYfull, 6, colors='k', linewidth=1)
plt.clabel(cs, inline=1, fontsize=10, colors='k')
ax2[1].set_title('Variance ($m^2$)')
for axx in ax2:
    axx.set_aspect('equal', 'datalim')
    axx.plot(sonar[dd,1], sonar[dd,2], '.', c=[.5, .5, .5], markersize=2)
    #axx.plot(V_Ireland[:,0], V_Ireland[:,1], 'k-')
    axx.set_xlabel('Easting ($m$)')
    axx.set_ylabel('Northing ($m$)')

plt.show()

