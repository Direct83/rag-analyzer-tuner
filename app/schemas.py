from pydantic import BaseModel
from typing import List, Dict, Optional

class SearchItem(BaseModel):
    score: float
    text: str
    filename: Optional[str] = None
    idx: Optional[int] = None

class SearchResponse(BaseModel):
    items: List[SearchItem]
    diagnostics: Dict[str, object]

class AskResponse(BaseModel):
    answer: str
    sources: List[SearchItem]
    diagnostics: Dict[str, object]

