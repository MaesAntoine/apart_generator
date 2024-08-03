import ghhops_server as hs
from simlation_graph import SimulationGraph
from block import Block

class SimulationGraphComponent:
    @staticmethod
    def register(hops):
        @hops.component(
            "/build_simulation_graph",
            name="Build Simulation Graph",
            nickname="buildSimuGraph",
            description="Build a Simulation Graph from a Grasshopper index tree and attributes",
            inputs=[
                hs.HopsInteger("Node indices", "Ni", "All node indices", hs.HopsParamAccess.TREE),
                hs.HopsString("Attributes", "Att", "All the attributes for a corresponding index", hs.HopsParamAccess.TREE),
                hs.HopsString("Labels", "L", "Labels for each attributes", hs.HopsParamAccess.LIST)
            ],
            outputs=[
                hs.HopsString("Simulation Graph", "SG", "The Simulation Graph as a JSON string")
            ]
        )
        def simulation_graph(node_indices, attributes, labels):
            print("ehy")
            simulation_graph = SimulationGraph()
            simulation_graph.add_nodes(node_indices, attributes, labels)
            return simulation_graph.get_graph()