import numpy as np
import GPy
import fast_marcher
import fm_graphtools

def zero_fun(a,b):
    return 0

class mat_cost_function:
    def __init__(self, graph, cost_fun=zero_fun, *args):
        self.mat = np.zeros((graph.width, graph.height))
        for x in range(graph.width):
            for y in range(graph.height):
                self.mat[x,y] = cost_fun(x,y, *args)
    
    def calc_cost(self,a,b):
        return self.mat[a,b]

class fast_marching_explorer:
    def __init__(self, gridsize, start_node, end_node, X, Y, mean_value=0, obs=[], corridor=False):
        self.start_node = start_node
        self.end_node = end_node
        
        # create simple GP model
        self.X = X
        self.Y = Y
        self.mean_value = mean_value
        self.GP_model = GPy.models.GPRegression(X,Y-mean_value,GPy.kern.RBF(2))
        self.GP_model.kern.lengthscale = 14
        self.GP_model.kern.variance = 45
        self.GP_model.Gaussian_noise.variance = 2.0
        
        # create cost graph from the GP estimate
        self.GP_cost_graph = fm_graphtools.CostmapGridFixedObs(gridsize[0], gridsize[1], obstacles=obs)
        Xtemp, Ytemp = np.meshgrid(np.arange(self.GP_cost_graph.width), np.arange(self.GP_cost_graph.height))
        self.Xfull = np.vstack([Xtemp.ravel(), Ytemp.ravel()]).transpose()
        self.Yfull, self.varYfull = self.GP_model.predict(self.Xfull)
        self.Yfull += mean_value
        self.cmodel = mat_cost_function(self.GP_cost_graph)
        self.cmodel.mat = np.reshape(self.Yfull, (self.GP_cost_graph.height, self.GP_cost_graph.width)).transpose()
        self.GP_cost_graph.cost_fun = self.cmodel.calc_cost
        
        self.fbFM = fast_marcher.FullBiFastMarcher(self.GP_cost_graph)
        self.fbFM.set_start(self.start_node)
        self.fbFM.set_goal(self.end_node)
        
        
    def cost_update(self, cost_update):
        self.fbFM.update(cost_update)
        return self.fbFM.updated_min_path_cost
    
    def cost_update_new(self, cost_update,loc=None):
        self.fbFM.update_new2(cost_update,loc)
        return self.fbFM.updated_min_path_cost
        
    def add_observation(self, Xnew, Ynew):
        self.X = np.append(self.X, [Xnew], axis=0)
        self.Y = np.append(self.Y, [[Ynew]], axis=0)
        self.GP_model.set_XY(self.X, self.Y-self.mean_value)
        
        self.Yfull, self.varYfull = self.GP_model.predict(self.Xfull)
        self.Yfull += self.mean_value
        self.cmodel.mat = np.reshape(self.Yfull, (self.GP_cost_graph.height, self.GP_cost_graph.width)).transpose()
        self.GP_cost_graph.cost_fun = self.cmodel.calc_cost
        self.GP_cost_graph.clear_delta_costs()
        
        self.fbFM.set_graph(self.GP_cost_graph)
        self.fbFM.set_start(self.start_node)
        self.fbFM.set_goal(self.end_node)
        self.fbFM.search()
        self.fbFM.pull_path()

    def set_plots(self, imf, ax):
        self.fbFM.set_plots(imf, ax)
        
    def set_plot_costs(self, startcost, delta_cost):
        self.fbFM.set_plot_costs(startcost, delta_cost)
        
    def find_corridor(self):
        self.fbFM.find_corridor()
        
    def search(self):
        # Initial search
        self.fbFM.search()
        self.fbFM.pull_path()