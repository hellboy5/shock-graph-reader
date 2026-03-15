from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass(frozen=True)
class SamplePoint:
    """Immutable representation of a spatial-temporal shock point."""
    sample_id: int
    x: float
    y: float
    t: float
    speed: float
    theta: float
    phi: float

class Node:
    """A point in the Shock DAG with GNN-ready connectivity metrics."""
    def __init__(self, node_id: int, node_type: str):
        self.id = node_id
        self.type = node_type  # SOURCE, SINK, JUNCT, etc.
        self.sample: Optional[SamplePoint] = None
        self._cw_neighbors: List[int] = []
        
        # Directed edge tracking
        self.incoming_edges: List['Edge'] = []
        self.outgoing_edges: List['Edge'] = []

    def add_neighbor(self, neighbor_id: int):
        """Adds an adjacent node ID in clockwise order."""
        self._cw_neighbors.append(neighbor_id)

    def get_cw_neighbors(self) -> List[int]:
        """Returns the list of adjacent node IDs."""
        return self._cw_neighbors

    @property
    def in_degree(self) -> int:
        """Number of incoming directed edges."""
        return len(self.incoming_edges)

    @property
    def out_degree(self) -> int:
        """Number of outgoing directed edges."""
        return len(self.outgoing_edges)

    @property
    def degree(self) -> int:
        """Total number of connected edges (in + out)."""
        return self.in_degree + self.out_degree

class Edge:
    """A directed branch in the DAG connecting source to target."""
    def __init__(self, edge_id: int, source: Node, target: Node, 
                 samples: List[SamplePoint]):
        self.id = edge_id
        self.source = source
        self.target = target
        self.samples = samples
        
        # Automatic connectivity registration
        self.source.outgoing_edges.append(self)
        self.target.incoming_edges.append(self)

class ShockGraph:
    """The logical container for the Shock Graph DAG."""
    def __init__(self):
        self.metadata: Dict[str, str] = {}
        self.nodes: Dict[int, Node] = {}
        self.edges: List[Edge] = []

