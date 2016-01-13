import fm_graphtools
import fm_plottools
import math
import numpy as np
import copy

def zero_heuristic(a, b):
    return 0

def FM_reset(FM_new, FM_old, graph):
    graph.delta_costs = {}
    FM_new.set_graph(graph)
    FM_new.frontier = copy.copy(FM_old.frontier)
    FM_new.cost_to_come = copy.copy(FM_old.cost_to_come)
    FM_new.parent_list = copy.copy(FM_old.parent_list)
    FM_new.child_list = copy.copy(FM_old.child_list)
    FM_new.downwind_nodes = 0
    FM_new.search_nodes = 0

def bFM_reset(FM_new, FM_old, graph):
    FM_reset(FM_new, FM_old, graph)
    FM_new.best_midnode = copy.copy(FM_old.best_midnode)
    FM_new.best_midfacenode = copy.copy(FM_old.best_midfacenode)
    FM_new.best_cost = copy.copy(FM_old.best_cost)
    FM_new.global_parent = copy.copy(FM_old.global_parent)
    
class FastMarcher:
    adjacency_list = {(-1,0): [(0,1), (0,-1)], (1,0): [(0,1), (0,-1)], 
        (0,-1): [(1,0), (-1,0)], (0,1):  [(1,0), (-1,0)]}
        
    def __init__(self, graph):
        self.graph = graph
        self.image_frames = 0
        self.axes = 0
        self.heuristic_fun = zero_heuristic
        self.delta_plot = 1
        self.plot_cost = 1
        self.step_dist = 0.21
        self.downwind_nodes = 0
        self.upwind_nodes = 0
        self.search_nodes = 0
    
    def set_graph(self, graph):
        self.graph = graph
        
    def set_start(self, node):
        self.start_node = node
        
    def set_goal(self, node):
        self.end_node = node
        
    def set_plots(self, imf, ax):
        self.image_frames = imf
        self.axes = ax
        
    def set_plot_costs(self, startcost, delta_cost):
        self.plot_cost = startcost
        self.delta_plot = delta_cost
        
    def set_heuristic(self, hfun):
        self.heuristic_fun = hfun
        
    def search(self):
        # Note that the nodes in the cost_to_come list are 'accepted'
        # Nodes in the queue are active but not accepted yet
        self.frontier = fm_graphtools.PriorityQueue([])
        self.frontier.clear()
        self.frontier.push(self.start_node, 0+self.heuristic_fun(self.start_node, self.end_node))
        self.cost_to_come = {}
        self.parent_list = {self.start_node:[self.start_node]}       
        
        self.continue_FM_search()
        self.create_child_list()

        
        # return self.cost_to_come, self.parent_list
    
    def continue_FM_search(self):
        nodes_popped=0
        if self.image_frames != 0:
            self.plot_cost, NULL = min(self.frontier.elements)
        
        while True:
            try:
                c_priority, c_node = self.frontier.pop()
            except KeyError:
                break
            nodes_popped+=1
            u_A = c_priority - self.heuristic_fun(c_node, self.end_node)
            self.cost_to_come[c_node] = u_A
            for n_node, tau_k in self.graph.neighbours(c_node):
    #            if n_node not in cost_to_come
                parent_update = []
                u_B = u_A + tau_k + 1.0
                adjacency = (n_node[0]-c_node[0], n_node[1]-c_node[1])
                for adjacent_node in self.adjacency_list[adjacency]:
                    B_node = (n_node[0]+adjacent_node[0], n_node[1]+adjacent_node[1])
                    if B_node in self.cost_to_come and self.cost_to_come[B_node] < u_B:
                        u_B = self.cost_to_come[B_node]
                        BB_node = B_node
                if tau_k > abs(u_A - u_B):
                    c_cost = 0.5*(u_A + u_B + math.sqrt(2*tau_k**2 - (u_A - u_B)**2))
                    parent_update = [c_node, BB_node]
                else:
                    if u_A <= u_B:
                        c_cost = u_A + tau_k
                        parent_update = [c_node]
                    else:
                        c_cost = u_B + tau_k
                        parent_update = [BB_node]
                if n_node not in self.cost_to_come or self.cost_to_come[n_node] > c_cost:
                    self.frontier.push(n_node, c_cost + self.heuristic_fun(n_node, self.end_node))
                    self.parent_list[n_node] = parent_update
                    
            if self.image_frames != 0 and u_A > self.plot_cost :            
                self.image_frames.append(fm_plottools.draw_costmap(self.axes, self.graph, self.cost_to_come, start_nodes=self.start_node))
                self.plot_cost += self.delta_plot
                
            if (c_node == self.end_node) or (self.frontier.count() <= 0):
                # self.frontier.push(c_node, u_A)
                break
    
        # Append final frame
        if self.image_frames != 0:
            self.image_frames.append(fm_plottools.draw_costmap(self.axes, self.graph, self.cost_to_come))
        # print "FM search: nodes popped: {0}".format(nodes_popped)
        self.search_nodes = nodes_popped
        # return self.cost_to_come, self.parent_list
      
    def pull_path(self):
        self.path = self.path_source_to_point(self.start_node, self.end_node)

    def path_source_to_point(self, source, target):
        path = [target]
        current_node = target
        while math.sqrt((current_node[0]-source[0])**2 + 
        (current_node[1]-source[1])**2) > 3+2*self.step_dist:
            if current_node in self.cost_to_come:
                grad = self.local_gradient(current_node)
            else:
                grad = np.array([0.0,0.0])
                weight = 0
                node_list = self.find_nearest_nodes(current_node)
                for row in node_list:
                    c_node = (row[0], row[1])
                    l_weight = 1/row[2]
                    grad += self.local_gradient(c_node)*l_weight
                    weight += l_weight
                grad = grad/weight
            
            if all( (t == 0 for t in grad) ):
                grad = np.random.randn(2)*0.1
            Mgrad = math.sqrt(grad[0]**2 + grad[1]**2)
            new_node = (current_node[0]+self.step_dist*grad[0]/Mgrad, current_node[1]+self.step_dist*grad[1]/Mgrad) 
            current_node = new_node
            path.append(current_node)
        path.append(source)
        path.reverse()
        return path        
    
    def find_nearest_nodes(self, node):
        dx = node[0]%1
        dy = node[1]%1
        fx = int(node[0])
        fy = int(node[1])
        node_list= []
        square = [[0,0], [0,1], [1,1], [1,0]]
        for offset in square:
            c_node = (fx+offset[0], fy+offset[1])
            if c_node in self.cost_to_come:
                node_list.append([c_node[0], c_node[1], math.sqrt((dx+offset[0]*(1-2*dx))**2 + (dy+offset[1]*(1-2*dy))**2)])
        return node_list
        
    def local_gradient(self, current_node):
        (x, y) = current_node
        current_cost = self.cost_to_come[current_node]
        (dx,dy) = (0,0)
        if (x+1,y) in self.cost_to_come and self.cost_to_come[(x+1,y)] < current_cost:
            dx = current_cost - self.cost_to_come[(x+1,y)]
        if (x-1,y) in self.cost_to_come and current_cost - dx > self.cost_to_come[(x-1,y)]:
            dx = self.cost_to_come[(x-1,y)] - current_cost
        if (x,y+1) in self.cost_to_come and self.cost_to_come[(x,y+1)] < current_cost:
            dy = current_cost - self.cost_to_come[(x,y+1)]
        if (x,y-1) in self.cost_to_come and current_cost - dy > self.cost_to_come[(x,y-1)]:
            dy = self.cost_to_come[(x,y-1)] - current_cost
        return np.array([dx, dy])

    def create_child_list(self):
        self.child_list = {}
        for child_node in self.parent_list:
            for parent in self.parent_list[child_node]:
                if parent not in self.child_list:
                    self.child_list[parent] = [child_node]
                else:
                    self.child_list[parent].append(child_node)
               
    def find_corridor(self):
        self.corridor, self.corridor_interface = \
            self.find_upwind(self.start_node, self.end_node)
            
    def find_upwind(self, source, target):
        nodes_popped = 0
        upwind_list = []
        upwind_interface = []
        search_list = fm_graphtools.PriorityQueue([])
        search_list.clear()
        c_node = target
        end_cost = self.cost_to_come[target]
        # Start from the end node, and note that the start_node must be a parent of
        # the end node (the true start node is everyone's parent, but other nodes 
        # may not be
        while c_node != source:
            upwind_list.append(c_node)
            for node, costTEMP in self.graph.neighbours(c_node):
                if node in self.parent_list[c_node]:
                    if node not in upwind_list:
                        search_list.push(node, end_cost - self.cost_to_come[node])
                elif node not in upwind_list:
                    upwind_interface.append(node)
            try:
                c_cost, c_node = search_list.pop()
                nodes_popped+=1
            except KeyError:
                break
        upwind_list.append(source)
        upwind_list.reverse()
        # print "FM upwind: nodes popped: {0}".format(nodes_popped)
        self.upwind_nodes = nodes_popped
        return upwind_list, upwind_interface
    
    def find_downwind(self, start_nodes):
        downwind_list = set([])
        interface_list = set([])
        search_list = fm_graphtools.PriorityQueue([])
        nodes_popped = 0
        for c_node in start_nodes:
            search_list.push(c_node, self.cost_to_come[c_node])
        
        c_cost, c_node = search_list.pop()
        # Start from the start node
        while True:
            downwind_list.update( {c_node} )
            for node, TEMPCOST in self.graph.neighbours(c_node):
                if (c_node in self.child_list) and node in self.child_list[c_node]:
                    if node in self.cost_to_come:
                        search_list.push(node, self.cost_to_come[node])
                elif node not in downwind_list:
                    interface_list.update( {node} )
            try:
                c_cost, c_node = search_list.pop()
                nodes_popped+=1
            except KeyError:
                break
                
        # Filter out nodes that are in the start set (don't want them in interface)
        interface_list = [enode for enode in interface_list if enode not in start_nodes]
        # print "FM downwind: nodes popped: {0}".format(nodes_popped)
        self.downwind_nodes = nodes_popped
        return downwind_list, interface_list
        
    def update(self, new_cost, force_update=False):
        # New cost should be added as a dictionary, with elements  [(node) : delta_cost]
        self.search_nodes = 0
        self.downwind_nodes = 0
        
        # Strip zero cost nodes
        new_cost = {node:new_cost[node] for node in new_cost if new_cost[node] != 0 and node in self.cost_to_come}
    
        # Check if all the changes aren't cost increases outside of the corridor (else return)
        change_corridor = False
        for node in new_cost:
            if (new_cost[node] < 0) or ((new_cost[node] > 0) and (node in self.corridor)):
                change_corridor = True;
                break   
        if (not force_update) and (not change_corridor):
            print "Only cost increases outside best corridor, returning"
            return
        
        self.graph.add_delta_costs(new_cost)
    
        # Add all non-zero cost changes to the kill list, and add their parents to the search list
        kill_list = set(new_cost.keys())
        interface = set()
        for node in new_cost:
            interface.update(self.parent_list[node])
    
        self.frontier.clear()
        #self.frontier.push(self.end_node, self.cost_to_come[self.end_node])
        cost_up_nodes = [node for node in new_cost if new_cost[node] > 0]
    
        # Find all nodes downwind of a cost-increased node, and add to the kill list
        if len(cost_up_nodes) > 0:
            new_kills, new_interface = self.find_downwind(cost_up_nodes)
            kill_list.update(new_kills)
            interface.update(new_interface)
        
        for internode in interface:
            if (internode in self.cost_to_come) and (internode not in kill_list): 
                self.frontier.push(internode, self.cost_to_come[internode])
        
        for killnode in kill_list:
            if killnode in self.cost_to_come: del self.cost_to_come[killnode]
            # if killnode in new_parent_list: del new_parent_list[killnode]
        
        self.continue_FM_search()
            
    def make_video(self, leading_frame=[], trailing_frame=[]):
        graph_frame, TEMP = fm_plottools.draw_grid(self.axes, self.graph)
        costpath_frame = fm_plottools.draw_costmap(self.axes, self.graph, self.cost_to_come, self.path)
        corridor_frame = fm_plottools.draw_corridor(self.axes, self.graph, self.cost_to_come, self.corridor, self.corridor_interface, self.path)
        path_frame, TEMP = fm_plottools.draw_grid(self.axes, self.graph, self.path)
        
        video_frames = []
        frame_hold = int(len(self.image_frames)/8)
        if len(leading_frame) > 0:
            for ii in range(frame_hold): video_frames.append(leading_frame)
        for ii in range(frame_hold): video_frames.append(graph_frame)
        video_frames.extend(self.image_frames)
        for ii in range(frame_hold): video_frames.append(costpath_frame)
        for ii in range(frame_hold): video_frames.append(corridor_frame)
        for ii in range(frame_hold): video_frames.append(path_frame)
        if len(trailing_frame) > 0:
            for ii in range(frame_hold): video_frames.append(trailing_frame)
        return video_frames
        
    def make_pictures(self, fig_dir):
        self.axes.figure.set_size_inches(9,6)
        graph_frame, barlims = fm_plottools.draw_grid(self.axes, self.graph)
        barlims = np.floor(barlims)
        delta = np.floor((barlims[1]-barlims[0])/6*10)/10
        cbar = self.axes.figure.colorbar(graph_frame[0], ticks=[barlims[0]+x*delta for x in range(7)])
        self.axes.figure.savefig(fig_dir+'graph_cost.pdf', bbox_inches='tight')
        
        path_frame, barlims  = fm_plottools.draw_grid(self.axes, self.graph, self.path)
        self.axes.figure.savefig(fig_dir+'graph_path.pdf', bbox_inches='tight')
       
        cbar.ax.clear()
        costpath_frame = fm_plottools.draw_costmap(self.axes, self.graph, self.cost_to_come, self.path)
        barlims = [min(self.cost_to_come.values()), max(self.cost_to_come.values())]
        delta = np.floor((barlims[1]-barlims[0])/6*10)/10
        self.axes.figure.colorbar(costpath_frame[0], ticks=[barlims[0]+x*delta for x in range(7)], cax=cbar.ax)
        self.axes.figure.savefig(fig_dir+'path_cost.pdf', bbox_inches='tight')                
        
        corridor_frame = fm_plottools.draw_corridor(self.axes, self.graph, self.cost_to_come, self.corridor, self.corridor_interface, self.path)
        self.axes.figure.savefig(fig_dir+'corridor.pdf', bbox_inches='tight')
       

class BiFastMarcher(FastMarcher):
    
    def search(self):
        # Note that the nodes in the cost_to_come list are 'accepted'
        # Nodes in the queue are active but not accepted yet
        self.frontier = fm_graphtools.PriorityQueue([])
        self.frontier.clear()
        self.frontier.push(self.start_node, 0)
        self.frontier.push(self.end_node, 0)
        
        self.global_parent = {self.start_node:self.start_node, self.end_node:self.end_node}
        self.cost_to_come = {}
        self.parent_list = {}
        
        self.continue_bFM_search()
        self.create_child_list()

        # return self.cost_to_come, self.parent_list
    
    def continue_bFM_search(self):        
        if self.image_frames != 0:
            self.plot_cost, NULL = min(self.frontier.elements)
        
        finished = False
        nodes_popped = 0

        while (self.frontier.count() > 0) and finished == False:
            try:
                c_priority, c_node = self.frontier.pop()
                if self.global_parent[c_node] != c_node:
                    for node in self.parent_list[c_node]:
                        if node not in self.cost_to_come:
                            raise ValueError("InvalidNode")
                nodes_popped+=1
            except ValueError:
                continue
            except KeyError:
                break
            u_A = c_priority
            self.cost_to_come[c_node] = u_A
            for n_node, tau_k in self.graph.neighbours(c_node):
    #            if n_node not in cost_to_come
                parent_update = []
                u_B = u_A + tau_k + 1.0
                adjacency = (n_node[0]-c_node[0], n_node[1]-c_node[1])
                for adjacent_node in self.adjacency_list[adjacency]:
                    B_node = (n_node[0]+adjacent_node[0], n_node[1]+adjacent_node[1])
                    if (B_node in self.cost_to_come) and \
                        (self.cost_to_come[B_node] < u_B) and \
                        (self.global_parent[B_node] == self.global_parent[c_node]):
                        u_B = self.cost_to_come[B_node]
                        BB_node = B_node
                if tau_k > abs(u_A - u_B):
                    c_cost = 0.5*(u_A + u_B + math.sqrt(2*tau_k**2 - (u_A - u_B)**2))
                    parent_update = [c_node, BB_node]
                else:
                    if u_A <= u_B:
                        c_cost = u_A + tau_k
                        parent_update = [c_node]
                    else:
                        c_cost = u_B + tau_k
                        parent_update = [BB_node]
                if n_node not in self.cost_to_come :
                    self.frontier.push(n_node, c_cost)
                    self.parent_list[n_node] = parent_update
                    self.global_parent[n_node] = self.global_parent[c_node]
                elif self.global_parent[n_node] == self.global_parent[c_node] and self.cost_to_come[n_node] > c_cost:
                    self.frontier.push(n_node, c_cost)
                    self.parent_list[n_node] = parent_update
                elif self.global_parent[n_node] != self.global_parent[c_node]:
                    self.best_midnode = c_node
                    self.best_midfacenode = n_node
                    self.best_cost = c_cost + self.cost_to_come[n_node]
                    self.frontier.push(c_node, u_A)
                    finished = True                                
            if self.image_frames != 0 and u_A > self.plot_cost :            
                self.image_frames.append(fm_plottools.draw_costmap(self.axes, self.graph, self.cost_to_come))
                self.plot_cost += self.delta_plot
                
        if self.image_frames != 0:
            self.image_frames.append(fm_plottools.draw_costmap(self.axes, self.graph, self.cost_to_come))
        # print "biFM search: nodes popped: {0}".format(nodes_popped)
        self.search_nodes=nodes_popped
        
            
    def pull_path(self):
        path1 = self.path_source_to_point(self.global_parent[self.best_midnode], self.best_midnode)
        path2 = self.path_source_to_point(self.global_parent[self.best_midfacenode], self.best_midfacenode)
        path1.remove(self.best_midnode)
        path2.remove(self.best_midfacenode)
        
        if self.global_parent[self.best_midnode] == self.start_node:
            path2.reverse()
            path1.extend(path2)
            self.path = path1
        else:
            path1.reverse()
            path2.extend(path1)
            self.path = path2
        
    def find_corridor(self):
        c1, ci1 = self.find_upwind(self.global_parent[self.best_midnode], self.best_midnode)
        temp = self.upwind_nodes
        c2, ci2 = self.find_upwind(self.global_parent[self.best_midfacenode], self.best_midfacenode)
        self.upwind_nodes+=temp
        
        c1.extend(c2)
        ci1.extend(ci2)
        self.corridor = c1
        self.corridor_interface = ci1
            
    def local_gradient(self, current_node):
        (x, y) = current_node
        current_cost = self.cost_to_come[current_node]
        g_parent = self.global_parent[current_node]
        (dx,dy) = (0,0)
        if (x+1,y) in self.cost_to_come and self.global_parent[(x+1,y)] == g_parent and self.cost_to_come[(x+1,y)] < current_cost:
            dx = current_cost - self.cost_to_come[(x+1,y)]
        if (x-1,y) in self.cost_to_come and self.global_parent[(x-1,y)] == g_parent and current_cost - dx > self.cost_to_come[(x-1,y)]:
            dx = self.cost_to_come[(x-1,y)] - current_cost
        if (x,y+1) in self.cost_to_come and self.global_parent[(x,y+1)] == g_parent and self.cost_to_come[(x,y+1)] < current_cost:
            dy = current_cost - self.cost_to_come[(x,y+1)]
        if (x,y-1) in self.cost_to_come and self.global_parent[(x,y-1)] == g_parent and current_cost - dy > self.cost_to_come[(x,y-1)]:
            dy = self.cost_to_come[(x,y-1)] - current_cost
        return np.array([dx, dy])
        
    def update(self, new_cost):
        # New cost should be added as a dictionary, with elements  [(node) : delta_cost]
        self.search_nodes = 0
        self.downwind_nodes = 0
        
        # Strip zero cost nodes
        new_cost = {node:new_cost[node] for node in new_cost if new_cost[node] != 0}
        self.graph.add_delta_costs(new_cost)
        
        temp_cost = {node:new_cost[node] for node in new_cost if (node in self.cost_to_come)}
        
        # Check if all the changes aren't cost increases outside of the corridor (else return)
        change_corridor = False
        for node in temp_cost:
            if (temp_cost[node] < 0) or ((temp_cost[node] > 0) and (node in self.corridor)):
                change_corridor = True;
                break   
        if not change_corridor:
            # print "Only cost increases outside best corridor, returning"
            return
    
        # Add all non-zero cost changes to the kill list, and add their parents to the search list
        kill_list = set(temp_cost.keys())
        interface = set()
        #for node in self.bifront_interface:
        #    if self.global_parent[node] != source:
        #        interface.add(node)
        for node in temp_cost:
            if node in self.parent_list:
                interface.update(self.parent_list[node])
    
        # self.frontier.clear()
        cost_up_nodes = [node for node in temp_cost if temp_cost[node] > 0]
    
        # Find all nodes downwind of a cost-increased node, and add to the kill list
        if len(cost_up_nodes) > 0:
            new_kills, new_interface = self.find_downwind(cost_up_nodes)
            kill_list.update(new_kills)
            interface.update(new_interface)
        
        for internode in interface:
            if (internode in self.cost_to_come) and (internode not in kill_list): 
                self.frontier.push(internode, self.cost_to_come[internode])
        
        for killnode in kill_list:
            if killnode in self.cost_to_come: del self.cost_to_come[killnode]
            # if killnode in new_parent_list: del new_parent_list[killnode]
        
        if self.image_frames != 0:
            self.axes.clear()
            fm_plottools.draw_corridor(self.axes, self.graph, self.cost_to_come, self.cost_to_come.keys(), interface-kill_list)
            self.axes.figure.savefig('/home/nick/Dropbox/work/FastMarching/fig/map_update.pdf', bbox_inches='tight')
        
        self.continue_bFM_search()


class FullBiFastMarcher:
    def __init__(self, graph):
        self.FastMarcherSG = FastMarcher(graph)
        self.FastMarcherGS = FastMarcher(graph)
        self.graph = graph
        self.image_frames = 0
        self.axes = 0
        
    def set_graph(self, graph):
        self.FastMarcherSG.graph = graph
        self.FastMarcherGS.graph = graph
        self.graph = graph
        
    def set_start(self, node):
        self.FastMarcherSG.set_start(node)
        self.FastMarcherSG.set_goal((None,None))
        self.start_node = node
        
    def set_goal(self, node):
        self.FastMarcherGS.set_start(node)
        self.FastMarcherGS.set_goal((None,None))
        self.end_node = node
        
    def set_plots(self, imf, ax):
        self.image_frames = imf
        self.axes = ax
        self.FastMarcherSG.set_plots(imf,ax)
        self.FastMarcherGS.set_plots(imf,ax)
        
    def set_plot_costs(self, startcost, delta_cost):
        self.plot_cost = startcost
        self.delta_plot = delta_cost
        self.FastMarcherSG.set_plot_costs(startcost, delta_cost)
        self.FastMarcherGS.set_plot_costs(startcost, delta_cost)
        
    def search(self):
        self.FastMarcherSG.search()
        self.FastMarcherGS.search()
        self.path_cost = copy.copy(self.FastMarcherSG.cost_to_come)
        for node in self.path_cost:
            self.path_cost[node] += self.FastMarcherGS.cost_to_come[node]
        self.FastMarcherGS.set_goal(self.FastMarcherSG.start_node)
        self.FastMarcherSG.set_goal(self.FastMarcherGS.start_node)
        self.min_path_cost = self.FastMarcherSG.cost_to_come[self.end_node]
            
    def pull_path(self, step=0.21):
        self.FastMarcherSG.step_dist = step
        self.FastMarcherSG.pull_path()
        self.path = self.FastMarcherSG.path

    def find_corridor(self):
        self.FastMarcherSG.find_corridor()
        self.corridor = self.FastMarcherSG.corridor
                
    def update(self, new_cost, recalc_path = 0):
        # New cost should be added as a dictionary, with elements  [(node) : delta_cost]
        midnode = (-1,-1)
        
        # Strip zero cost nodes and obstacle nodes
        new_cost = {node:new_cost[node] for node in new_cost if ((new_cost[node] != 0) and (node in self.path_cost))}
        self.graph.clear_delta_costs()
        self.graph.add_delta_costs(new_cost)
        
        # All nodes that have changed cost are killed
        kill_list = set(new_cost.keys())
        
        # Interface nodes are neighbours of a killed node with a lower cost (that are not also being killed)
        interface = set()
        cost_to_come = copy.copy(self.FastMarcherSG.cost_to_come)
        for node in new_cost:
            for nnode, costTEMP in self.graph.neighbours(node):
                if (cost_to_come[nnode] < cost_to_come[node]):
                    interface.add(nnode)
    
        # Find boundary points closest to start and goal
        min_cts = self.min_path_cost
        min_ctg = self.min_path_cost
        
        frontier = fm_graphtools.PriorityQueue([])
        frontier.clear()
        if self.start_node in kill_list:
            frontier.push(self.start_node, 0)
        
        # Push all interface nodes onto the frontier
        for internode in interface:
            if (internode in self.path_cost) and (internode not in kill_list): 
                frontier.push(internode, cost_to_come[internode])
                min_cts = min(min_cts, cost_to_come[internode])
                min_ctg = min(min_ctg, self.FastMarcherGS.cost_to_come[internode])
        if min_cts+min_ctg > self.min_path_cost:
            self.updated_min_path_cost = self.min_path_cost
            return
        
        
        temp_path_cost = copy.copy(self.path_cost)
        for killnode in kill_list:
            del cost_to_come[killnode]
            del temp_path_cost[killnode]
            # if killnode in new_parent_list: del new_parent_list[killnode]
        
        # Current minimum path cost
        self.updated_min_path_cost = min(temp_path_cost.values())
        #if self.image_frames != 0:
        #    self.axes.clear()
        #    fm_plottools.draw_corridor(self.axes, self.graph, self.cost_to_come, self.cost_to_come.keys(), interface-kill_list)
        #    self.axes.figure.savefig('/home/nick/Dropbox/work/FastMarching/fig/map_update.pdf', bbox_inches='tight')
        
        nodes_popped=0
        if self.image_frames != 0:
            self.plot_cost, NULL = min(frontier.elements)
        u_A = 0
        
        # I changed this from u_A + min_ctg < self.min_path_cost
        while u_A + min_ctg < self.min_path_cost:
            try:
                c_priority, c_node = frontier.pop()
            except KeyError:
                break
            nodes_popped+=1
            u_A = c_priority
            cost_to_come[c_node] = u_A
            for n_node, tau_k in self.graph.neighbours(c_node):
                u_B = u_A + tau_k + 1.0
                adjacency = (n_node[0]-c_node[0], n_node[1]-c_node[1])
                for adjacent_node in self.FastMarcherSG.adjacency_list[adjacency]:
                    B_node = (n_node[0]+adjacent_node[0], n_node[1]+adjacent_node[1])
                    if B_node in cost_to_come and cost_to_come[B_node] < u_B:
                        u_B = cost_to_come[B_node]
                        
                if tau_k > abs(u_A - u_B):
                    c_cost = 0.5*(u_A + u_B + math.sqrt(2*tau_k**2 - (u_A - u_B)**2))
                else:
                    if u_A <= u_B:
                        c_cost = u_A + tau_k
                    else:
                        c_cost = u_B + tau_k
                
                if n_node not in cost_to_come:
                    frontier.push(n_node, c_cost)
                elif n_node in interface:
                    if c_cost + self.FastMarcherGS.cost_to_come[n_node] < self.updated_min_path_cost:
                        self.updated_min_path_cost = c_cost + self.FastMarcherGS.cost_to_come[n_node]
                        midnode = c_node
                        #print "Better path found!"                
                    
            if self.image_frames != 0 and u_A > self.plot_cost :            
                self.image_frames.append(fm_plottools.draw_costmap(self.axes, self.graph, cost_to_come))
                self.plot_cost = u_A + self.delta_plot
            
    
        # Append final frame
        if self.image_frames != 0:
            self.image_frames.append(fm_plottools.draw_costmap(self.axes, self.graph, cost_to_come))
        # print "fbFM update: nodes popped: {0}".format(nodes_popped)
        self.search_nodes = nodes_popped
        
        if recalc_path:
            if midnode != (-1,-1):
                tempctc = copy.copy(self.FastMarcherSG.cost_to_come)
                self.FastMarcherSG.cost_to_come = cost_to_come
                path1 = self.FastMarcherSG.path_source_to_point(self.start_node, midnode)
                path2 = self.FastMarcherGS.path_source_to_point(self.end_node, midnode)
                path2.remove(midnode)
            
                path2.reverse()
                path1.extend(path2)
                self.updated_path = path1
                self.FastMarcherSG.cost_to_come = tempctc
            else:
                self.updated_path = copy.copy(self.path)

        return 

    def kill_downwind(self, start_nodes, cost_to_come, min_ctg):
        interface_list = set([])
        downwind_list = set([])
        search_list = fm_graphtools.PriorityQueue([])
        for c_node in start_nodes:
            search_list.push(c_node, self.FastMarcherSG.cost_to_come[c_node])
        
        c_cost, c_node = search_list.pop()
        # Start from the start node
        while True:
            downwind_list.update( {c_node} )
            del cost_to_come[c_node]
            # Check all the neighbours of the current node
            for node, TEMPCOST in self.graph.neighbours(c_node):
                # If they're children, add them to the search
                if (c_node in self.FastMarcherSG.child_list) and node in self.FastMarcherSG.child_list[c_node]:
                    # If we're still inside the min_ctg, keep adding nodes
                    if self.FastMarcherGS.cost_to_come[node] >= min_ctg:
                        search_list.push(node, self.FastMarcherSG.cost_to_come[node])
                # If they're not children and they're not in the downwind list, they're interface nodes
                elif node not in downwind_list:
                    interface_list.update( {node} )
            try:
                c_cost, c_node = search_list.pop()
                self.nodes_popped+=1
            except KeyError:
                break
        return interface_list
                  
    def update_new(self, new_cost, recalc_path = 0):
        # New cost should be added as a dictionary, with elements  [(node) : delta_cost]
        
        M_s = set()
        #M_g = set()
        P_g = {}
        A = {}
        
        Q = fm_graphtools.PriorityQueue([])
        Q.clear()
        
        A_s = copy.copy(self.FastMarcherSG.cost_to_come)
        A_g = copy.copy(self.FastMarcherGS.cost_to_come)
        
        min_cts = self.min_path_cost
        min_ctg = self.min_path_cost
        
        for x_d in new_cost:
            min_cts = min(A_s[x_d], min_cts)
            min_ctg = min(A_g[x_d], min_ctg)
            if new_cost[x_d] > 0:
                A_s[x_d] = self.min_path_cost
                A_g[x_d] = self.min_path_cost
            for x_p in self.FastMarcherSG.parent_list[x_d]:
                if x_p not in new_cost:
                    M_s.add(x_p)
            for x_p in self.FastMarcherGS.parent_list[x_d]:
                if x_p not in new_cost:
                    M_s.add(x_p)

        if min_cts + min_ctg > self.min_path_cost:
            self.updated_min_path_cost = self.min_path_cost
            return
                                  
        for x_M in M_s:
            c_s = A_s[x_M]-min_cts
            c_g = A_g[x_p]-min_ctg
            if c_s < c_g:
                Q.push(x_M, c_s)
                P_g[x_M] = False
                for x_p in self.FastMarcherSG.parent_list[x_M]:
                    A[x_p] = A_s[x_p]-min_cts
                
        #for x_M in M_g:
        #    Q.push(x_M, A_g[x_M]-min_ctg)
        #    P_g[x_M] = True
        #    for x_p in self.FastMarcherGS.parent_list[x_M]:
        #        A[x_p] = A_g[x_p]-min_ctg       
        
        C_prime = min([A_s[x] + A_g[x] for x in A_s])
        C_b = C_prime - min_cts - min_ctg

        if C_b < 0:
            self.updated_min_path_cost = self.min_path_cost
            return
            
        C_up = self.ReSearch(Q, A, P_g, C_b)
        self.updated_min_path_cost = C_up + min_cts + min_ctg
        
    def ReSearch(self, Q, A, P_g, C_b):
        c_n = 0
        nodes_popped = 0
        while (Q.count() > 0) and (c_n < C_b/2):
            try:
                c_n, n = Q.pop()
                nodes_popped+=1
            except KeyError:
                break
                
            if n in A:
                return c_n + A[n]
            A[n] = c_n
            
            for m, tau in self.graph.neighbours(n):                    
                c_k = c_n + tau + 1.0
                for k, temp in self.graph.neighbours(m):
                    if k in A and A[k] < c_k:
                        c_k = A[k]
                        
                if tau > abs(c_n - c_k):
                    c_cost = 0.5*(c_n + c_k + math.sqrt(2*tau**2 - (c_n - c_k)**2))
                else:
                    if c_n <= c_k:
                        c_cost = c_n + tau
                    else:
                        c_cost = c_k + tau
                
                if m in A and P_g[m] < P_g[n]:
                    if c_cost < A[m]:
                        del A[m]
                        Q.push(m, c_cost)
                        P_g[m] = P_g[n]
                else:
                    Q.push(m, c_cost)
                    P_g[m] = P_g[n]
                    
            if self.image_frames != 0 and c_n > self.plot_cost :            
                self.image_frames.append(fm_plottools.draw_costmap(self.axes, self.graph, A))
                self.plot_cost = c_n + self.delta_plot
            
                
        if self.image_frames != 0:
            self.image_frames.append(fm_plottools.draw_costmap(self.axes, self.graph, A))
        print "biFM ReSearch: nodes popped: {0}".format(nodes_popped)
        self.search_nodes=nodes_popped    

        return C_b

    def update_new2(self, new_cost,loc=None):
        # New cost should be added as a dictionary, with elements  [(node) : delta_cost]
        self.nodes_popped = 0

        self.graph.clear_delta_costs()
        self.graph.add_delta_costs(new_cost)
        
        Q = fm_graphtools.PriorityQueue([])
        Q.clear()
        
        if self.image_frames != 0:
            self.image_frames.append(self.plot_cost_frame(self.FastMarcherSG.cost_to_come,loc))
        
        min_cts = self.min_path_cost
        min_ctg = self.min_path_cost
                
        for x_d in new_cost:
            min_cts = min(self.FastMarcherSG.cost_to_come[x_d], min_cts)
            min_ctg = min(self.FastMarcherGS.cost_to_come[x_d], min_ctg)

        if min_cts + min_ctg > self.min_path_cost:
            self.updated_min_path_cost = self.min_path_cost
            return
        
        A_prime = copy.copy(self.FastMarcherSG.cost_to_come)
        if (new_cost.itervalues().next() > 0):
            M = self.kill_downwind(new_cost.keys(), A_prime, min_ctg)
        else:
            M = set()
            for x_d in new_cost:
                for x_p in (y for y in self.FastMarcherSG.parent_list[x_d] if y not in new_cost):
                    M.update({x_p})
        if self.FastMarcherSG.start_node in new_cost:
            M.update({self.FastMarcherSG.start_node})

        for x_M in M:
            Q.push(x_M, self.FastMarcherSG.cost_to_come[x_M])
                
        C_prime = min([A_prime[x] + self.FastMarcherGS.cost_to_come[x] for x in A_prime])
        C_b = C_prime - min_ctg

        if min_cts > C_b:
            self.updated_min_path_cost = C_prime
            return
            
        C_prime = self.ReSearch2(Q, A_prime, C_b, min_ctg,loc)
        self.updated_min_path_cost = C_prime                

    def ReSearch2(self, frontier, cost_to_come, C, min_ctg,loc=None):
        if self.image_frames != 0:
            self.plot_cost, NULL = min(frontier.elements)
        
        while True:
            try:
                c_priority, c_node = frontier.pop()
            except KeyError:
                break
            self.nodes_popped+=1
            u_A = c_priority
            if u_A > C:
                if self.image_frames != 0:
                    self.image_frames.append(self.plot_cost_frame(cost_to_come, loc))
                return C+min_ctg
            elif self.FastMarcherGS.cost_to_come[c_node] <= min_ctg:
                if self.image_frames != 0:
                    self.image_frames.append(self.plot_cost_frame(cost_to_come, loc))
                return u_A + self.FastMarcherGS.cost_to_come[c_node]
                    
            cost_to_come[c_node] = u_A
            for n_node, tau_k in self.graph.neighbours(c_node):
                u_B = u_A + tau_k + 1.0
                adjacency = (n_node[0]-c_node[0], n_node[1]-c_node[1])
                for adjacent_node in self.FastMarcherSG.adjacency_list[adjacency]:
                    B_node = (n_node[0]+adjacent_node[0], n_node[1]+adjacent_node[1])
                    if B_node in cost_to_come and cost_to_come[B_node] < u_B:
                        u_B = cost_to_come[B_node]
                if tau_k > abs(u_A - u_B):
                    c_cost = 0.5*(u_A + u_B + math.sqrt(2*tau_k**2 - (u_A - u_B)**2))
                else:
                    if u_A <= u_B:
                        c_cost = u_A + tau_k
                    else:
                        c_cost = u_B + tau_k
                if n_node not in cost_to_come or cost_to_come[n_node] > c_cost:
                    frontier.push(n_node, c_cost)
                    
            if self.image_frames != 0 and u_A > self.plot_cost :            
                self.image_frames.append(self.plot_cost_frame(cost_to_come, loc))
                self.plot_cost += self.delta_plot
    
    def plot_cost_frame(self, cost_to_come,loc):
        tempframe=fm_plottools.draw_costmap(self.axes, self.FastMarcherSG.graph, cost_to_come)
        if loc != None:
            tempframe.append(self.axes.plot(loc[0], loc[1], 'wx', mew=2, ms=10)[0])
            tempframe.append(self.axes.plot(loc[0], loc[1], 'wo', mew=1, ms=80, mfc='none', mec='w' )[0])
        return tempframe
    

'''
        min_cost = 1000
        for n_node, tau_k in self.graph.neighbours(self.best_midnode):
            if (self.global_parent[n_node] != self.global_parent[self.best_midnode]) and \
                (self.cost_to_come[n_node] < min_cost):
                self.best_midfacenode = n_node
                min_cost = tau_k
                
                
        # Do the updates for each half of the search (from start and from goal)
        self.update_half(new_cost, self.start_node)
        self.update_half(new_cost, self.end_node)
        self.continue_bFM_search()

    def update_half(self, new_cost, source):
        
        temp_cost = {node:new_cost[node] for node in new_cost if ((node in self.cost_to_come) and (self.global_parent[node] == source))}
        
        # Check if all the changes aren't cost increases outside of the corridor (else return)
        change_corridor = False
        for node in temp_cost:
            if (temp_cost[node] < 0) or ((temp_cost[node] > 0) and (node in self.corridor)):
                change_corridor = True;
                break   
        if not change_corridor:
            print "Only cost increases outside best corridor, returning"
            return
    
        # Add all non-zero cost changes to the kill list, and add their parents to the search list
        kill_list = set(new_cost.keys())
        interface = set()
        #for node in self.bifront_interface:
        #    if self.global_parent[node] != source:
        #        interface.add(node)
        for node in new_cost:
            interface.update(self.parent_list[node])
    
        # self.frontier.clear()
        cost_up_nodes = [node for node in new_cost if new_cost[node] > 0]
    
        # Find all nodes downwind of a cost-increased node, and add to the kill list
        if len(cost_up_nodes) > 0:
            new_kills, new_interface = self.find_downwind(cost_up_nodes)
            kill_list.update(new_kills)
            interface.update(new_interface)
        
        for internode in interface:
            if (internode in self.cost_to_come) and (internode not in kill_list): 
                self.frontier.push(internode, self.cost_to_come[internode])
        
        for killnode in kill_list:
            if killnode in self.cost_to_come: del self.cost_to_come[killnode]
            # if killnode in new_parent_list: del new_parent_list[killnode]


'''