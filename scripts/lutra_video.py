import pickle
import numpy as np
import GPy
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import scipy

fh = open('/home/nick/Lutra/data/timed_log_trial2.p', 'rb')
samples = pickle.load(fh)
poses = pickle.load(fh)
targets = pickle.load(fh)
fh.close()
gridsize = [90, 80]
mean_value = 3

pond_img=scipy.misc.imread('/home/nick/Lutra/data/junction_city_pond.png')
pond_size = [227,194,3]
pond_origin = [-75, -40]
new_pond_img = np.empty(pond_size, dtype=np.uint8)

iX = 0
while samples[iX,0] == 0:
    iX+=1
X = samples[0:iX,1:3]
Y = np.array([samples[0:iX,3]]).transpose()
GP_model = GPy.models.GPRegression(X,Y-mean_value,GPy.kern.RBF(2))
GP_model.kern.lengthscale = 16
GP_model.kern.variance = 25
GP_model.Gaussian_noise.variance = 0.5

iT = 0
target_pos = targets[iT,[1,2]]
next_target = targets[iT+1,0]

fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.set_position([0.05, 0.05, 0.9, 0.9])
ax.imshow(pond_img, extent = (pond_origin[0],pond_origin[0]+pond_size[0] ,pond_origin[1], pond_origin[1]+pond_size[1]))
ax.set_xlim([-10, pond_origin[0]+pond_size[0]]); 
ax.set_ylim([-10, pond_origin[1]+pond_size[1]]); 
cmap = plt.cm.terrain

lframes = []
ctime = 0
Xtemp, Ytemp = np.meshgrid(np.arange(gridsize[0]), np.arange(gridsize[1]))
Xfull = np.vstack([Xtemp.ravel(), Ytemp.ravel()]).transpose()
Yfull, varYfull = GP_model.predict(Xfull)
Yfull += mean_value
Ymat = np.reshape(Yfull, (gridsize[1], gridsize[0]))
next_sample = samples[iX+1,0]

#def init():
l_poses = ax.plot(poses[0,1], poses[0,2], 'k-')[0]
l_targets = ax.plot(targets[0,1], targets[0,2], 'bo')[0]
l_samples = ax.plot(X[:,0], X[:,1], 'rx')[0]
m_GP = ax.matshow(Ymat, interpolation='none', cmap=cmap, vmin=mean_value, vmax=28, alpha=0.5)
plot_objects = [l_poses, l_targets, l_samples, m_GP]

    #return [l_poses, l_targets, l_samples, m_GP]

def update_plot(ii, l_poses, l_targets, l_samples, m_GP):
    global next_target, next_sample, iT, iX
    ctime = poses[ii,0]
    if ctime > next_target:
        iT=min(iT+1, targets.shape[0]-2)
        next_target = targets[iT+1,0]
        l_targets.set_data(targets[iT,1], targets[iT,2])
    if ctime > next_sample:
        iX = min(iX+1, samples.shape[0]-2)
        next_sample = samples[iX+1,0]
        X = samples[0:iX+1,1:3]
        Y = np.array([samples[0:iX+1,3]]).transpose()
        GP_model.set_XY(X,Y-mean_value)
        Yfull, varYfull = GP_model.predict(Xfull)
        Yfull += mean_value
        Ymat = np.reshape(Yfull, (gridsize[1], gridsize[0]))
        l_samples.set_data(X[:,0], X[:,1])
        m_GP.set_data(Ymat)
   
    l_poses.set_data(poses[0:ii+1,1], poses[0:ii+1,2])
    return [l_poses, l_targets, l_samples, m_GP]
    
    
vid1 = ani.FuncAnimation(fig, update_plot, poses.shape[0], fargs=(plot_objects), interval=50)
vid1.save('/home/nick/Lutra/data/trial2_fivespeed.mp4', writer = 'avconv', fps=2, bitrate=1500)
plt.show()