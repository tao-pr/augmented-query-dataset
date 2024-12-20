from pydantic import BaseModel
from enum import Enum

# See readme (query augmentation section)
class VariantType(Enum):
    LEMMA = 1
    SYN_REPL = 2
    POS_REORD = 3
    SPELLING = 4
    HYPERNYM = 5
    HYPONYM = 6
    ACRONYM = 7
    NER_SYN = 8
    TRANSL = 9
    MODIF = 10

# https://platform.openai.com/docs/guides/structured-outputs
class QueryVariant(BaseModel):
    query: str
    lang: str
    variant_type: VariantType

class Query(BaseModel):
    original: str
    lang: str

class QuerySet(BaseModel):
    queries: list[Query]



    