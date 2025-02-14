from typing import Optional


class Relation:
    def __init__(
        self, source_id: str, target_id: str, label: str, metadata: Optional[str] = None
    ):
        self.source_id = source_id
        self.target_id = target_id
        self.label = label
        self.metadata = metadata
