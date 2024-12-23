from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Optional
import torch


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
    audio: Optional[torch.Tensor] = None
    audio_score: Optional[float] = None


@dataclass_json
@dataclass
class Segment:
    word: str
    start: float
    end: float
    score: Optional[float] = 0


@dataclass
class Sample:
    node_1: Node
    node_2: Node
    label: int
