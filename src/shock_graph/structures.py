"""Data structures representing a shock graph and its components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SamplePoint:
    """Immutable representation of a spatial-temporal shock point.

    Attributes:
        sample_id: The unique integer ID of the sample.
        x: The x-coordinate of the point.
        y: The y-coordinate of the point.
        t: The radius of the maximal inscribed circle.
        speed: The speed of the shock formation.
        theta: The tangent angle in radians.
        phi: The speed-derived angle in radians.
    """
    sample_id: int
    x: float
    y: float
    t: float
    speed: float
    theta: float
    phi: float


@dataclass
class EdgeShapeFeatures:
    """Stores the computed geometric features for an edge.

    Attributes:
        s_length: Arc-length of the shock edge.
        s_curve: Total absolute curvature of the shock edge.
        s_angle: Total absolute angle change of the shock edge.
        p_length: Arc-length of the plus (left) boundary.
        p_curve: Total absolute curvature of the plus boundary.
        p_angle: Total absolute angle change of the plus boundary.
        m_length: Arc-length of the minus (right) boundary.
        m_curve: Total absolute curvature of the minus boundary.
        m_angle: Total absolute angle change of the minus boundary.
        poly_area: Area of the polygon formed by the shock and boundaries.
    """
    s_length: float = 0.0
    s_curve: float = 0.0
    s_angle: float = 0.0
    p_length: float = 0.0
    p_curve: float = 0.0
    p_angle: float = 0.0
    m_length: float = 0.0
    m_curve: float = 0.0
    m_angle: float = 0.0
    poly_area: float = 0.0


class Node:
    """A point in the Shock DAG with GNN-ready connectivity metrics.

    Attributes:
        id: The unique integer identifier for the node.
        type: The string type of the node (e.g., 'SOURCE', 'SINK', 'JUNCT').
        sample: An optional SamplePoint associated directly with the node.
        incoming_edges: A list of directed edges targeting this node.
        outgoing_edges: A list of directed edges originating from this node.
    """

    def __init__(self, node_id: int, node_type: str) -> None:
        """Initializes a new Node instance.

        Args:
            node_id: The unique identifier for the node.
            node_type: The classification type of the node.
        """
        self.id = node_id
        self.type = node_type
        self.sample: Optional[SamplePoint] = None
        self._cw_neighbors: List[int] = []

        # Directed edge tracking
        self.incoming_edges: List[Edge] = []
        self.outgoing_edges: List[Edge] = []

    def add_neighbor(self, neighbor_id: int) -> None:
        """Adds an adjacent node ID in clockwise order.

        Args:
            neighbor_id: The ID of the neighboring node.
        """
        self._cw_neighbors.append(neighbor_id)

    def get_cw_neighbors(self) -> List[int]:
        """Returns the list of adjacent node IDs.

        Returns:
            A list of adjacent node IDs in clockwise order.
        """
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
    """A directed branch in the DAG connecting source to target.

    Attributes:
        id: The unique integer identifier for the edge.
        source: The originating Node.
        target: The destination Node.
        samples: A list of SamplePoint objects describing the shock curve.
        features: The computed geometric features for this edge.
    """

    def __init__(
        self,
        edge_id: int,
        source: Node,
        target: Node,
        samples: List[SamplePoint],
    ) -> None:
        """Initializes a new Edge instance.

        Args:
            edge_id: The unique identifier for the edge.
            source: The node where the edge originates.
            target: The node where the edge terminates.
            samples: The ordered list of sample points forming the edge.
        """
        self.id = edge_id
        self.source = source
        self.target = target
        self.samples = samples
        self.features: Optional[EdgeShapeFeatures] = None

        # Automatic connectivity registration
        self.source.outgoing_edges.append(self)
        self.target.incoming_edges.append(self)

    # --- Feature Accessors ---

    @property
    def s_length(self) -> float:
        """Returns the length of the shock curve."""
        return self.features.s_length if self.features else 0.0

    @property
    def s_curve(self) -> float:
        """Returns the total curvature of the shock curve."""
        return self.features.s_curve if self.features else 0.0

    @property
    def poly_area(self) -> float:
        """Returns the area of the polygon formed by the shock boundaries."""
        return self.features.poly_area if self.features else 0.0


class ShockGraph:
    """The logical container for the Shock Graph DAG.

    Attributes:
        metadata: A dictionary storing graph-level metadata.
        nodes: A dictionary mapping node IDs to Node objects.
        edges: A list of all Edge objects in the graph.
    """

    def __init__(self) -> None:
        """Initializes an empty ShockGraph instance."""
        self.metadata: Dict[str, str] = {}
        self.nodes: Dict[int, Node] = {}
        self.edges: List[Edge] = []
