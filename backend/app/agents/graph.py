from app.agents.state import AgentState
from app.agents.nodes.query_analysis import query_analysis_node
from app.agents.nodes.retrieval import retrieval_node
from app.agents.nodes.context_validation import context_validation_node
from app.agents.nodes.rewrite_on_failure import rewrite_query_node
from app.agents.nodes.generation import generation_node

END = "__END__"

class StateGraph:
    def __init__(self, state_schema):
        self.nodes = {}
        self.edges = {}
        self.conditional_edges = {}
        self.entry_point = None

    def add_node(self, name, func):
        self.nodes[name] = func

    def set_entry_point(self, name):
        self.entry_point = name

    def add_edge(self, from_node, to_node):
        self.edges[from_node] = to_node

    def add_conditional_edges(self, from_node, routing_func, route_map):
        self.conditional_edges[from_node] = (routing_func, route_map)

    def compile(self):
        class CompiledGraph:
            def __init__(self, nodes, edges, conditional_edges, entry_point):
                self.nodes = nodes
                self.edges = edges
                self.conditional_edges = conditional_edges
                self.entry_point = entry_point

            async def ainvoke(self, state, config=None, **kwargs):
                current_node = self.entry_point
                while current_node != END:
                    # Run node
                    func = self.nodes[current_node]
                    node_update = await func(state)
                    if node_update:
                        state.update(node_update)

                    # Route to next
                    if current_node in self.conditional_edges:
                        routing_func, route_map = self.conditional_edges[current_node]
                        route_val = routing_func(state)
                        current_node = route_map.get(route_val, END)
                    elif current_node in self.edges:
                        current_node = self.edges[current_node]
                    else:
                        current_node = END
                return state
        return CompiledGraph(self.nodes, self.edges, self.conditional_edges, self.entry_point)

def route_after_analysis(state: AgentState):
    if state.get("route") == "conversational":
        return "generate"
    return "retrieve"

def route_after_validation(state: AgentState):
    if state.get("needs_rewrite") and state.get("rewrite_count", 0) < 2:
        return "rewrite"
    return "generate"

# Initialize graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("analyze", query_analysis_node)
workflow.add_node("retrieve", retrieval_node)
workflow.add_node("validate", context_validation_node)
workflow.add_node("rewrite", rewrite_query_node)
workflow.add_node("generate", generation_node)

# Add edges
workflow.set_entry_point("analyze")

workflow.add_conditional_edges(
    "analyze",
    route_after_analysis,
    {
        "generate": "generate",
        "retrieve": "retrieve"
    }
)

workflow.add_edge("retrieve", "validate")

workflow.add_conditional_edges(
    "validate",
    route_after_validation,
    {
        "rewrite": "rewrite",
        "generate": "generate"
    }
)

workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

# Compile the graph
agent = workflow.compile()
