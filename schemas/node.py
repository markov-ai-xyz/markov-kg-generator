from typing import List, Optional


class ParentNode:
    def __init__(
        self,
        node_id: str,
        source_id: str,
        text: str,
        embedding: List[float],
        metadata: Optional[str] = None,
        start_char_idx: Optional[int] = None,
        end_char_idx: Optional[int] = None,
    ):
        self.node_id = node_id
        self.source_id = source_id
        self.text = text
        self.embedding = embedding
        self.metadata = metadata
        self.start_char_idx = start_char_idx
        self.end_char_idx = end_char_idx


class ChildNode:
    def __init__(
        self,
        triplet_source_id: str,
        text: str,
        embedding: List[float],
        metadata: Optional[str] = None,
        start_char_idx: Optional[int] = None,
        end_char_idx: Optional[int] = None,
    ):
        self.triplet_source_id = triplet_source_id
        self.text = text
        self.embedding = embedding
        self.metadata = metadata
        self.start_char_idx = start_char_idx
        self.end_char_idx = end_char_idx
