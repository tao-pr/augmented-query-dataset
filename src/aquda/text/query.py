from pydantic import BaseModel
from enum import Enum

# See readme (query augmentation section)
class VariantType(Enum):
    LEMMA = 'Lemmatization or word stemming'
    SYN_REPL = 'Synonym replacement'
    SPELLING = 'Spelling or Misspelling variants'
    HYPERNYM = 'Hypernym'
    HYPONYM = 'Hyponym'
    ACRONYM = 'Acronym expansion or collapse'
    NER_SYN = 'Synonym or similar term with Named-entity-recognition'
    TRANSL = 'Language translation'
    MODIF = 'Adding or replacement of common modifiers'

PARAM_MAP = {
    'lemma' : VariantType.LEMMA,
    'syn-repl': VariantType.SYN_REPL,
    'spelling': VariantType.SPELLING,
    'hypernym': VariantType.HYPERNYM,
    'hyponym': VariantType.HYPONYM,
    'acronym': VariantType.ACRONYM,
    'ner-syn': VariantType.NER_SYN,
    'transl': VariantType.TRANSL,
    'modif': VariantType.MODIF
}

PARAMS = list(PARAM_MAP.keys())

# https://platform.openai.com/docs/guides/structured-outputs
class Query(BaseModel):
    original: str
    lang: str

class QuerySet(BaseModel):
    queries: list[Query]

class VariantElement(BaseModel):
    text: str
    lang: str
    variant_type: VariantType

class QueryVariant(Query):
    variants: list[VariantElement]

class QueryVariantSet(BaseModel):
    queries: list[QueryVariant]



    