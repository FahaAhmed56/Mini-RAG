"""Text processing shared by every retriever. web/rag.js implements the exact same rules."""
import re

STOPWORDS = set("""a an the and or but if of to in on at by for with from as is are was were be been being am do does did
i me my we our you your he she it its they them their this that these those there here what which who whom when where why how
can could should would will shall may might must have has had not no yes so than then too very just also into about over
any all each every some such only own same other more most up out off again further once per get gets got""".split())


def stem(w):
    """A tiny suffix-stripping stemmer: 'apartments' -> 'apartment', 'charges' -> 'charg'... consistent, not pretty."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 3 and w.endswith("es") and w[-3] in "sxz":
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokenize(text):
    """lowercase -> words -> drop stopwords -> stem"""
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
