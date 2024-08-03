import networkx as nx
from typing import Dict, Any, List, Tuple

class Apartment:
    def __init__(self, apartment_id: int, blocks: List[Block]):
        self.id = apartment_id
        self.blocks = blocks

    def add_block(self, block: Block):
        self.blocks.append(block)

    def __repr__(self):
        return f"Apartment(id={self.id}, blocks={len(self.blocks)})"