import numpy as np
import GPy
import fast_marcher
import fm_graphtools

def zero_fun(a,b):
    return 0

class mat_cost_function:
    def __init__(self, graph, cost_fun=zero_fun, *args, **kwargs):
        self.mat = np.zeros((graph.width, graph.height))
        self.left = graph.left
        self.bottom = graph.bottom
        for x in range(graph.width):
            for y in range(graph.height):
                self.mat[x,y] = cost_fun(self.left+x, self.bottom+y, *args, **kwargs)
    
    def calc_cost(self,a,b):
        return self.mat[a-self.left,b-self.bottom]
        
class mat_cost_function_GP:
    def __init__(self, graph, cost_fun=zero_fun, *args, **kwargs):
        lx = np.arange(graph.left, graph.right, dtype='int')
        ly = np.arange(graph.bottom, graph.top, dtype='int')
        X_star = np.array([[x,y] for x in lx for y in ly])
        Y_star,Y_var = cost_fun(X_star[:,0], X_star[:,1],*args, **kwargs)
        self.cost_dict = {(X_star[k,0],X_star[k,1]):Y_star[k,0] for k in range(Y_star.shape[0])}
        self.var_dict  = {(X_star[k,0],X_star[k,1]):Y_var[k,0]  for k in range(Y_var.shape[0])}
    
    def calc_cost(self,a,b):
        return self.cost_dict[(a,b)]
        
    def calc_var(self,a,b):
        return self.var_dict[(a,b)]

def GP_cost_function(x, y, GPm, max_depth=1.0e3, mean_depth=0.0):
    # Cost function shold be strictly positive (depth < max_depth)
    X = np.array([np.ravel(x), np.ravel(y)]).transpose()
    mean,var = GPm.predict(X)
    mean = max_depth-(mean+mean_depth)
    mean[mean < 0.1] = 0.1
    if len(mean) == 1:
        return mean[0],var[0]
    else:
        return mean,var
            
class fast_marching_explorer:
    def __init__(self, gridsize, start_node, end_node, X, Y, mean_value=0, obs=[], GP_l=14.0,GP_sv=45.0,GP_sn=1.0,bl_corner=[0,0],*args,**kwargs):
        self.start_node = start_node
        self.end_node = end_node
        
        # create simple GP model
        self.X = X
        self.Y = Y
        self.mean_value = mean_value
        self.GP_model = GPy.models.GPRegression(X,Y-mean_value,GPy.kern.RBF(2))
        self.GP_model.kern.lengthscale = GP_l
        self.GP_model.kern.variance = GP_sv
        self.GP_model.Gaussian_noise.variance = GP_sn
        
        # create cost graph from the GP estimate
        self.GP_cost_args = args
        self.GP_cost_kwargs = kwargs        
        self.GP_cost_graph = fm_graphtools.CostmapGridFixedObs(gridsize[0], gridsize[1], obstacles=obs, bl_corner=bl_corner)
        self.cmodel = mat_cost_function_GP(self.GP_cost_graph, 
            cost_fun=GP_cost_function, GPm=self.GP_model, *self.GP_cost_args, **self.GP_cost_kwargs)
        self.GP_cost_graph.cost_fun = self.cmodel.calc_cost
        self.GP_cost_graph.var_fun = self.cmodel.calc_var
                
        #Xtemp, Ytemp = np.meshgrid(np.arange(self.GP_cost_graph.width), np.arange(self.GP_cost_graph.height))
        #self.Xfull = np.vstack([Xtemp.ravel(), Ytemp.ravel()]).transpose()
        #self.Yfull, self.varYfull = self.GP_model.predict(self.Xfull)
        #self.Yfull += mean_value
        #self.cmodel = mat_cost_function(self.GP_cost_graph)
        #self.cmodel.mat = np.reshape(self.Yfull, (self.GP_cost_graph.height, self.GP_cost_graph.width)).transpose()
        #self.GP_cost_graph.cost_fun = self.cmodel.calc_cost
        
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
        
        self.cmodel = mat_cost_function_GP(self.GP_cost_graph, 
            cost_fun=GP_cost_function, GPm=self.GP_model, *self.GP_cost_args, **self.GP_cost_kwargs)
        self.GP_cost_graph.cost_fun = self.cmodel.calc_cost
        self.GP_cost_graph.var_fun = self.cmodel.calc_var

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