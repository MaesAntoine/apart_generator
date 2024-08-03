import networkx as nx
from typing import Dict, Any, List, Tuple

class Block:
    def __init__(self, block_id: Any, attributes: Dict[str, Any] = None):
        self.id = block_id
        self.attributes = attributes or {}

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Block):
            return self.id == other.id
        return False

    def __repr__(self):
        return f"Block(id={self.id}, attributes={self.attributes})"