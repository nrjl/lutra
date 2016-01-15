import heapq
import math

# Primarily from here:
# http://www.redblobgames.com/pathfinding/a-star/implementation.html
# PriortyQueue from:
# https://docs.python.org/2/library/heapq.html#basic-examples

def unit_cost_function(a, b):
    return 1
    
def blob_cost_function(a, b):
    cost = 1 
    cost += 2*math.exp(-math.sqrt((a-40)**2 + (b-40)**2)/16)
    cost += 1*math.exp(-math.sqrt((a-110)**2 + (b-22)**2)/8)
    cost += 0.5*math.exp(-math.sqrt((a-110)**2 + (b-50)**2)/32)
    return cost
    
def euclidean_heuristic(node, end_node):
    return math.sqrt((node[0]-end_node[0])**2 + (node[1]-end_node[1])**2)

def manhattan_heuristic(node, end_node):
    return abs(node[0]-end_node[0]) + abs(node[1]-end_node[1])
       
def zero_heuristic(node, end_node):
    return 0

def square_cost_modifier(graph, xlo, xhi, ylo, yhi, delta):
    cost_dict={}
    for x in range(xlo, min(graph.width, xhi+1)):
        for y in range(ylo, min(graph.height, yhi+1)):
            if (x,y) not in graph.obstacles:
                cost_dict[(x,y)] = delta
    return cost_dict
    
def polynomial_cost_modifier(graph, cx, cy, r, delta):
    q = 1; D = 2
    j = D/2+q+1
    cost_dict={}
    for x in range(cx-r, min(graph.width, cx+r)):
        for y in range(cy-r, min(graph.height, cy+r)):
            if (x,y) not in graph.obstacles:
                dd = min(1, math.sqrt((x-cx)**2 + (y-cy)**2)/r)
                kd = delta*( max(0,(1-dd))**(j+1)*((j+1)*dd + 1) )
                if kd != 0: cost_dict[(x,y)] = kd
    return cost_dict
    
class polynomial_precompute_cost_modifier:
    def __init__(self, graph, r, min_val=0):
        self.graph = graph
        self.r = r
        self.min_val = min_val
        q = 1
        D = 2
        self.j = D/2+q+1
        self.build_dict()
        
    def build_dict(self):
        self.cost_dict={}
        for x in range(-self.r, self.r):
            for y in range(-self.r, self.r):
                dd = min(1, math.sqrt(x**2 + y**2)/self.r)
                kd = 1.0*( max(0,(1-dd))**(self.j+1)*((self.j+1)*dd + 1) )
                if kd > self.min_val: self.cost_dict[(x,y)] = kd
                
    def calc_cost(self, cx, cy, delta):
        out_cost = {(x+cx,y+cy):delta*self.cost_dict[(x,y)] for (x,y) in self.cost_dict 
            if x+cx >= self.graph.left and x+cx < self.graph.right and
                y+cy >= self.graph.bottom and y+cy < self.graph.top and
                (x+cx,y+cy) not in self.graph.obstacles}
        return out_cost
        
class poly_cost:
    def __init__(self, graph, r):
        q = 1; D = 2
        j = D/2+q+1
        self.cost_dict={}
        self.obs = graph.obstacles
        self.width = graph.width
        self.height = graph.height
        for x in range(-r, r+1):
            for y in range(-r, r+1):
                dd = min(1, math.sqrt((x)**2 + (y)**2)/r)
                kd = 1.0*( max(0,(1-dd))**(j+1)*((j+1)*dd + 1) )
                if kd != 0: self.cost_dict[(x,y)] = kd
                
    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height
            
    def passable(self, id):
        return id not in self.obs
                                        
    def set_update(self, cx, cy, delta):
        cx = int(round(cx))
        cy = int(round(cy))
        plist = [(x+cx, y+cy) for (x,y) in self.cost_dict]
        plist = filter(self.in_bounds, plist)
        plist = filter(self.passable, plist)
        out = {(x,y):delta*self.cost_dict[(x-cx,y-cy)] for (x,y) in plist}
        return out

class Graph:
    def __init__(self):
        self.edges = {}

    def neighbors(self, id):
        return self.edges[id]

class CostmapGrid:
    def __init__(self, width, height, cost_fun=unit_cost_function, obstacles=[], bl_corner=(0,0)):
        self.width = width
        self.height = height
        self.obstacles = obstacles         
        self.cost_fun = cost_fun
        self.delta_costs = {}
        self.bl_corner = bl_corner
        self.set_bounds()
        
    def set_bounds(self):
        self.left = self.bl_corner[0]
        self.right = self.bl_corner[0] + self.width
        self.bottom = self.bl_corner[1]
        self.top = self.bl_corner[1] + self.height
        
    def copy(self):
        return CostmapGrid(self.width, self.height, cost_fun=self.cost_fun, obstacles=self.obstacles, bl_corner=self.bl_corner)
        
    def in_bounds(self, id):
        (x, y) = id
        return self.left <= x < self.right and self.bottom <= y < self.top
   
    def passable(self, id):
        return id not in self.obstacles
   
    def neighbours(self, id):
        (x, y) = id
        results = [(x+1, y), (x, y-1), (x-1, y), (x, y+1)]
        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)
        results = [((a, b), self.node_cost((a, b))) for a,b in results]
        return results
      
    def node_cost(self, id):
        cost = self.cost_fun(id[0], id[1])
        if id in self.delta_costs:
            cost = cost+self.delta_costs[id]
        return max(0, cost)
        
    def add_delta_costs(self, cost_dictionary):
        self.delta_costs.update(cost_dictionary)
        
    def clear_delta_costs(self):
        self.delta_costs = {}
      
class PriorityQueue(object):
    # Priority queue based on the heap object
 
    def __init__(self, elements=[]):
        # if 'heap' already contains some items, heapify them
        heapq.heapify(elements)
        self.elements = elements
        
        # The entry_finder is used to identify duplicates. When you
        # want to add an element that already exists (reprioritise)
        # you just mark the old entry as invalid and push the new
        # one. Then when you pop you only pop valid entries.
        self.entry_finder = dict({i[-1]: i for i in elements})
        self.REMOVED = '<removed-node>'
 
    def push(self, node, priority=0):
        # If the new node is a duplicate of a current node (i.e. it 
        # appears in the entry_finder), delete the old one
        
        if node in self.entry_finder:
            self.delete(node)
        entry = [priority, node]
        self.entry_finder[node] = entry
        heapq.heappush(self.elements, entry)
 
    def delete(self, node):
        # When deleting a node, pop the element from the entry_finder
        # dictionary, and set the node value to the REMOVED keyword
        entry = self.entry_finder.pop(node)
        entry[-1] = self.REMOVED
        return entry[0]
 
    def pop(self):
        # When popping a node, pop the first one off the heapq. Check 
        # if it was deleted, if not, return it, otherwise keep poppin'
        while self.elements:
            priority, node = heapq.heappop(self.elements)
            if node is not self.REMOVED:
                del self.entry_finder[node]
                return priority, node
        raise KeyError('pop from an empty priority queue')
        
    def count(self):
        return len(self.elements)
        
    def clear(self):
        while len(self.elements) > 0:
            heapq.heappop(self.elements)

class CostmapGridFixedObs(CostmapGrid):
    def __init__(self, width, height, cost_fun=unit_cost_function, obstacles=[], bl_corner=(0,0)):
        self.width = width
        self.height = height
        self.obstacles = obstacles         
        self.cost_fun = cost_fun
        self.delta_costs = {}
        self.bl_corner=bl_corner
        self.set_bounds()
        self.rebuild_neighbours()
             
    def copy(self):
        return CostmapGridFixedObs(self.width, self.height, cost_fun=self.cost_fun, obstacles=self.obstacles, bl_corner=self.bl_corner)
   
    def passable(self, id):
        return id not in self.obstacles
   
    def neighbours(self, id):
        results = [(node, self.node_cost(node)) for node in self.fixed_neighbours[id]]
        return results
        
    def update_obstacles(self, new_obstacles):
        for obs in new_obstacles:
            if obs not in self.obstacles:
                self.obstacles.append(obs)
        self.rebuild_neighbours()
        
    def rebuild_neighbours(self):
        self.fixed_neighbours = {}
        for x in range(self.left, self.right):
            for y in range(self.bottom, self.top):
                results = [(x+1, y), (x, y-1), (x-1, y), (x, y+1)]
                results = filter(self.in_bounds, results)
                results = filter(self.passable, results)
                self.fixed_neighbours.update({(x,y):results})