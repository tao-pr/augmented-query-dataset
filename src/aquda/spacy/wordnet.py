import spacy
import numpy as np

from ..text import query

def create(model: str, lang: str, silence: bool) -> object:
    # Follow a convention from https://spacy.io/models#conventions
    model_name = f'{lang}_core_web_{model}'
    if not silence:
        print(f'Loading Spacy model: {model_name}')
    nlp = spacy.load(model_name)
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

def similar_of(nlp: object, tokens: set[str], num: int=1) -> dict[str, set[str]]:
    out = dict()

    # https://spacy.io/api/vectors#most_similar
    vectors = np.array([vector_of(nlp, t) for t in tokens]) # multiple tokens at once
    neighs = nlp.vocab.vectors.most_similar(vectors, n=num)
    for tok, nn in zip(tokens, neighs):
        # N nearest neighbours of token `tok`
        # NOTE: Not all the wordnet models are properly normalised.
        #        Therefore, the results might not always be linguistically accurate.
        nindexes, _, nscores = nn
        out[tok] = set([str_from_index(nlp, n).lower() for n in nindexes])
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

def synnonym_repl(lang: str, nlp: object, text: str) -> query.QueryVariant:
    # taotodo

    # POS tag + NER and replacement of other synnonyms
    pass

