import numpy as np  
from graphviz import Digraph 
import matplotlib.pyplot as plt
import networkx as nx


class Planner:
    def __init__(self, config, env):
        self.leaves = []
        self.nodes = []
        self.root = None 
        self.config = config
        self.env = env
        self.plan_path = None 
        self.logger = {'episode': [], 'return': []}

    def decompose_budgets(self):
        episode = 1
        for M in range(self.config['budget']):
            horizon = np.log(M)/(2*np.log(1/self.config['gamma']))
            if M*horizon <= self.config['budget']:
                episode = max(episode,M)
        
        horizon = int(np.log(episode)/(2*np.log(1/self.config['gamma'])))
        self.config['horizon'] = horizon 
        self.config['episode'] = episode 

    def simulation(self):
        self.env.reset()

        print(self.env.render())
        for action in self.plan_path:
            self.env.step(action)
            print('----------------------------------------')
            print(self.env.render())
            print('----------------------------------------')
            if np.all(self.env.x == self.env.goal):
                print('reached goal !!!!!!!!!! ')
                break 

    def visualize_tree(self):
        G = nx.DiGraph()
        
        root_id = "00"
        root_label = "Root\nstate: s0"
        G.add_node(root_id, label=root_label, color="lightblue")
        
        height = 0
        layer = {root_id: [node for node in self.root.children.values()]}
        
        while layer != {}:
            new_layer = {}
            idx = 1
            for parent_name in layer:
                for child in layer[parent_name]:
                    child_name = f"{height}{idx}"
                    
                    node_label = f"UCB: {child.mu:.2f}\nVU: {child.prefix() if child.value_sequence() == None else child.value_sequence() :.2f} \n act: {child.action} \n cout: {child.count}"
                    G.add_node(child_name, label=node_label, color="lightblue")
                    G.add_edge(parent_name, child_name)
                    
                    if child.children:
                        new_layer[child_name] = [c for c in child.children.values() if c.count > 0]
                    idx += 1
            height += 1
            layer = new_layer

        levels = {"00": 0}
        for node in G.nodes():
            if node != "00":
                levels[node] = len(nx.shortest_path(G, "00", node)) - 1


        nx.set_node_attributes(G, levels, "subset")
        pos = nx.multipartite_layout(G, subset_key= "subset")
        pos = {node: (y, -x) for node, (x, y) in pos.items()}
        scale_x = 0.3
        scale_y = 2

        for k, (x, y) in pos.items():
            pos[k] = (x * scale_x, y * scale_y)

        node_labels = nx.get_node_attributes(G, 'label')
        node_colors = [G.nodes[n].get('color', 'lightgray') for n in G.nodes()]
        
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=900, edgecolors="black")
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=5, verticalalignment="center")
        
        plt.axis("off")
        plt.show()
    