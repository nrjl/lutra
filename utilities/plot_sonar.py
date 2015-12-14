import pickle
import numpy as np
import GPy
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

fh = open('/home/nick/Lutra/data/sonar_data.pkl', 'rb')
poses = pickle.load(fh)
sonar = pickle.load(fh)
temps = pickle.load(fh)
fh.close()

dd = sonar[:,3] < 10.0
dd[0:375] = False

print "Building GP model..."
X = sonar[dd,1:3]
y = np.atleast_2d(sonar[dd,3]).transpose()
ymean = y.mean()
GP_model = GPy.models.GPRegression(X,y-ymean,GPy.kern.RBF(2))
GP_model.kern.lengthscale = 20
GP_model.kern.variance = 10
GP_model.Gaussian_noise.variance = 0.25
print "Optimizing..."
GP_model.optimize()
print "Done."

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

plt.show()