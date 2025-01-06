import spacy
import numpy as np

from ..text import query
from ..cli import colour

def create(model: str, silence: bool) -> object:
    if not silence:
        print(f'{colour.CYAN}Loading Spacy model:{colour.DEFAULT} {model}')
    nlp = spacy.load(model)
    return nlp

def get_spacer(lang: str) -> str:
    if lang.lower() in {'chinese', 'japanese', 'korean'}:
        return ''
    else:
        return ' '
    
def vector_of(nlp: object, token: str) -> np.array:
    return nlp(token).vector

def index_of(nlp: object, token: str) -> int:
    return nlp.vocab.strings[token]

def str_from_index(nlp: object, idx: int) -> str:
    return nlp.vocab.strings[idx]

def similar_of(nlp: object, tokens: set[str], num: int=2) -> dict[str, set[str]]:
    out = dict()

    # https://spacy.io/api/vectors#most_similar
    vectors = np.array([vector_of(nlp, t) for t in tokens]) # multiple tokens at once
    neighs = nlp.vocab.vectors.most_similar(vectors, n=num)

    nindexes, _, nscores = neighs
    for token, nindex, nscore in zip(tokens, nindexes, nscores):
        # N nearest neighbours of token `token`
        # NOTE: Not all models are properly normalised. This may
        #       deteriorate the quality of the vector similarity.
        out[token] = set(str_from_index(nlp, index).lower() for index in nindex)

    return out

def ner(nlp: object, text: str) -> list[str]:
    return list(nlp(text).ents)

def lemmatize(lang: str, nlp: object, text: str) -> query.QueryVariant:
    doc = nlp(text)
    lemma = get_spacer(lang).join([token.lemma_ for token in doc])
    return query.QueryVariant(
        original = text,
        lang = lang,
        variants = [
            query.VariantElement(
                text = lemma,
                lang = lang,
                variant_type = query.VariantType.LEMMA
            )
        ]
    )

def synonym_repl(lang: str, nlp: object, text: str) -> query.QueryVariant:
    doc = nlp(text)
    words = list(token.text for token in doc)
    similar_words: dict[str, list[str]] = similar_of(nlp, words, num=4)

    # Clean similar words
    # - Lemmatize
    # - Remove duplicates
    def clean(w: str, swords: list[str]) -> list[str]:
        return list(set(sw.lemma_ for sw in nlp(get_spacer(lang).join(swords)) if sw.lemma_ != w))

    similar_words = { w: clean(w, swords) for w, swords in similar_words.items() }
    # taotodo here

    # POS tag + NER and replacement of other synnonyms
    pass

