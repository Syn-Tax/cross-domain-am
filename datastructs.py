from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Optional


@dataclass_json
@dataclass
class Relation:
    type: str
    to_node_id: int


@dataclass_json
@dataclass
class Node:
    id: Optional[int] = None
    locution: Optional[str] = None
    proposition: Optional[str] = None
    relations: Optional[list[Relation]] = None


@dataclass_json
@dataclass
class WordSpan:
    word: str
    start: float
    end: float
