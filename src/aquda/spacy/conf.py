from pydantic import BaseModel
from typing import Optional

class LangConf(BaseModel):
    model: str

class Conf(BaseModel):
    langs: dict[str, LangConf]

