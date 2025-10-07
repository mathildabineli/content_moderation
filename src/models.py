from typing import List

from pydantic import BaseModel


class PredictInput(BaseModel):
    texts: List[str]
