import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)

def init_fig(graph=None, **kwargs):
    fign=plt.figure( **kwargs)
    axn=fign.add_subplot(111)
    axn.set_aspect('equal')
    axn.tick_params(labelbottom='on',labeltop='off')
    axn.set_xlabel(r'x')
    axn.set_ylabel(r'y') 
    if graph == None:
        axn.autoscale(tight=True)
    else:
        axn.set_xlim(-0.5, graph.width-.5)
        axn.set_ylim(-0.5, graph.height-0.5)
    return fign, axn
    
def clear_content(frame):
    for item in frame:
            item.remove()

def generate_obstacles(width, height, num_obstacles, max_size=None):
    if max_size==None:
        max_size = width/4
    obs_list = []
    for ii in range(num_obstacles):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        sx = random.randint(1, min(max_size, width-x))
        sy = random.randint(1, min(max_size, height-y))
        obs_list.extend([(a, b) for a in range(x, x+sx) for b in range(y, y+sy)])
    return obs_list
            
def draw_grid(axes, grid, path=None, max_cost = 0, min_cost = None, cost_fun=None, *args, **kwargs):
    if cost_fun == None:
        cost_fun = grid.node_cost
    grid_mat = np.zeros((grid.width, grid.height))
    for x in range(grid.width):
        for y in range(grid.height):
            grid_mat[x,y] = cost_fun((grid.left+x, grid.bottom+y))
    for x,y in grid.obstacles:
        grid_mat[x-grid.left,y-grid.bottom] = -1
    grid_mat = np.ma.masked_where(grid_mat == -1, grid_mat)
    max_cost = max(max_cost, grid_mat.max())
    if min_cost == None:
        min_cost = grid_mat.min()
    cmap = plt.cm.terrain
    cmap.set_bad(color='black')
    axes.set_xlim([grid.left, grid.right]); axes.set_ylim([grid.bottom, grid.top])
    mat_out =  [axes.imshow(grid_mat.transpose(), origin='lower', extent=[grid.left,grid.right,grid.bottom,grid.top],
        interpolation='none', cmap=cmap, vmin=min_cost, vmax=max_cost, *args, **kwargs)]
    if not path == None:
        x, y = zip(*path)
        mat_out.append(axes.plot(x, y, 'w-', linewidth=2.0 )[0])
        mat_out.append(axes.plot(x[0], y[0], 'r^', markersize=8 )[0])
        mat_out.append(axes.plot(x[-1], y[-1], 'ro', markersize=8 )[0])
    axes.tick_params(labelbottom='on',labeltop='off')
    return mat_out, [min_cost, max_cost]
    
def draw_costmap(axes, grid, cost_to_come, path=[], start_nodes=None):
    cost_mat = -1*np.ones((grid.width, grid.height))
    for node in cost_to_come:
        cost_mat[node[0]-grid.left,node[1]-grid.bottom] = cost_to_come[node]
    for x,y in grid.obstacles:
        cost_mat[x-grid.left,y-grid.bottom] = -2
    cost_mat = np.ma.masked_where(cost_mat == -2, cost_mat)
    cmap = plt.cm.jet
    cmap.set_bad(color='black')
    cmap.set_over(color='#C2A366')
    cmap.set_under(color='0.8')

    axes.set_xlim([grid.left, grid.right]); axes.set_ylim([grid.bottom, grid.top])
    mat_out = [axes.imshow(cost_mat.transpose(), origin='lower', extent=[grid.left,grid.right,grid.bottom,grid.top],
        norm = matplotlib.colors.Normalize(vmin=0, vmax=cost_mat.max(), clip=False))]
    if start_nodes != None:
        mat_out.append(axes.plot(start_nodes[0], start_nodes[1],'r^', markersize=8 )[0])
    if len(path) > 0:
        x, y = zip(*path)
        mat_out.append(axes.plot(x, y, 'w-', linewidth=2.0 )[0])
    axes.tick_params(labelbottom='on',labeltop='off')
    return mat_out


def draw_corridor(axes, grid, cost_to_come, corridor, interface=[], path=[]):
    cost_mat = -1*np.ones((grid.width, grid.height))
    for node in corridor:
        cost_mat[node[0]-grid.left,node[1]-grid.bottom] = cost_to_come[node]
    for x,y in grid.obstacles:
        cost_mat[x-grid.left,y-grid.bottom] = -2
    max_cost = cost_mat.max()
    for node in interface:
        cost_mat[node[0]-grid.left,node[1]-grid.bottom] = max_cost+1
    cost_mat = np.ma.masked_where(cost_mat == -2, cost_mat)
    cmap = plt.cm.jet
    cmap.set_bad(color='black')
    cmap.set_over(color='#C2A366')
    cmap.set_under(color='0.8')

    axes.set_xlim([grid.left, grid.right]); axes.set_ylim([grid.bottom, grid.top])
    mat_out = [axes.imshow(cost_mat.transpose(), origin='lower',extent=[grid.left,grid.right,grid.bottom,grid.top],
        norm = matplotlib.colors.Normalize(vmin=0, vmax=max_cost, clip=False))]
    if len(path) > 0:
        x, y = zip(*path)
        mat_out.append(axes.plot(x, y, 'w-', linewidth=2.0 )[0])
    axes.tick_params(labelbottom='on',labeltop='off')

    return mat_out


def draw_fbfmcost(axes, grid, path_cost, path=[], min_cost = 1e7, max_cost = 0):
    grid_mat = np.zeros((grid.width, grid.height))
    for x in range(grid.left, grid.right):
        for y in range(grid.bottom, grid.top):
            if (x,y) in path_cost:
                grid_mat[x-grid.left,y-grid.bottom] = path_cost[(x,y)]
    for x,y in grid.obstacles:
        grid_mat[x-grid.left,y-grid.bottom] = -1
    grid_mat = np.ma.masked_where(grid_mat == -1, grid_mat)
    cmap = plt.cm.jet
    cmap.set_bad(color='black')
    axes.set_xlim([grid.left, grid.right]); axes.set_ylim([grid.bottom, grid.top])
    max_cost = max(max_cost, grid_mat.max())
    min_cost = min(min_cost, min(path_cost.values()))
    mat_out =  [axes.imshow(grid_mat.transpose(), origin='lower', extent=[grid.left,grid.right,grid.bottom,grid.top],
        interpolation='none', cmap=cmap, vmax=max_cost, vmin=min_cost)]
    if len(path) > 0:
        x, y = zip(*path)
        mat_out.append(axes.plot(x, y, 'w-', linewidth=2.0 )[0])
    axes.tick_params(labelbottom='on',labeltop='off')
    #axes.figure.colorbar(mat_out[0])
    return mat_out, [grid_mat.min(), grid_mat.max()]