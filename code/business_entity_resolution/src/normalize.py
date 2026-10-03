"""Text normalisation for business names and addresses.

Everything here is language/country agnostic at the pipeline level: rules are
applied to every record regardless of its country label, so unseen countries
(e.g. France in the test set) are handled by the same code path.
"""
import re
from functools import lru_cache

from unidecode import unidecode

# ---------------------------------------------------------------- names
# canonical short forms (long form -> short form) so both spellings collide
NAME_CANON = {
    "corporation": "corp", "corpn": "corp", "incorporated": "inc", "company": "co",
    "limited": "ltd", "ltda": "ltd", "private": "pvt", "pvt": "pvt", "pte": "pvt",
    "international": "intl", "internatl": "intl", "manufacturing": "mfg",
    "services": "svc", "service": "svc", "svcs": "svc", "brothers": "bros",
    "associates": "assoc", "association": "assoc", "industries": "ind", "industry": "ind",
    "enterprises": "ent", "enterprise": "ent", "national": "natl", "management": "mgmt",
    "group": "grp", "technologies": "tech", "technology": "tech", "hospital": "hosp",
    "laboratories": "labs", "laboratory": "labs", "lab": "labs", "trading": "trdg",
    "traders": "trdrs", "solutions": "soln", "solution": "soln", "systems": "sys",
    "system": "sys", "products": "prod", "product": "prod", "center": "ctr",
    "centre": "ctr", "department": "dept", "university": "univ", "institute": "inst",
    "saint": "st", "sainte": "ste", "societe": "ste", "compagnie": "cie",
    "etablissements": "ets", "and": "", "et": "", "und": "", "y": "",
    # US / generic
    "incorp": "inc", "incorporation": "inc", "corporations": "corp", "companies": "co",
    "holdings": "hldgs", "holding": "hldgs", "partners": "ptnrs", "partner": "ptnrs",
    # India
    "prvt": "pvt", "privat": "pvt", "lmtd": "ltd", "limted": "ltd", "limitd": "ltd",
    # France
    "etablissement": "ets", "freres": "bros", "groupe": "grp",
}
# transliterated legal words (Hindi/Tamil/Gujarati/... script -> unidecode, e.g.
# "praaivett limittedd", "piraiveett limittett") are matched by their consonant
# skeleton, which is script-independent: praaivett/piraiveett -> prvt -> pvt
LEGAL_SKEL = {"prvt": "pvt", "prbt": "pvt", "prvr": "pvt",
              "lmtd": "ltd", "lmtt": "ltd", "lmrd": "ltd", "ltd": "ltd",
              "ellp": "llp", "elelp": "llp"}
# tokens that carry little identity (legal forms, fillers); removed for "core" name
LEGAL = {
    "inc", "corp", "co", "ltd", "llc", "llp", "lp", "pvt", "plc", "the", "of", "pllc",
    "sarl", "sas", "sa", "sasu", "eurl", "sci", "snc", "gmbh", "ag", "bv", "nv", "opc",
    "le", "la", "les", "de", "du", "des", "d", "l", "cie", "ste", "ets", "dba", "aka",
    "t/a", "ta", "india", "usa", "us", "france",
    # more legal forms: US professional corporations, French EI/SELARL/SCP/..., misc
    "pc", "psc", "lc", "ei", "eirl", "selarl", "selas", "scp", "scm", "gie", "sca",
    "sarlu", "scop", "sccv",
}

# ------------------------------------------------------------- addresses
ADDR_CANON = {
    "street": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave", "avn": "ave",
    "boulevard": "blvd", "bd": "blvd", "blv": "blvd", "drive": "dr", "lane": "ln",
    "court": "ct", "highway": "hwy", "parkway": "pkwy", "place": "pl", "square": "sq",
    "suite": "ste", "apartment": "apt", "floor": "fl", "building": "bldg", "bldng": "bldg",
    "north": "n", "south": "s", "east": "e", "west": "w", "near": "nr", "opposite": "opp",
    "opp.": "opp", "behind": "bhd", "number": "no", "mount": "mt", "fort": "ft",
    "circle": "cir", "terrace": "ter", "plaza": "plz", "route": "rte", "chemin": "ch",
    "impasse": "imp", "allee": "all", "faubourg": "fbg", "quai": "qu", "cedex": "",
    "saint": "st", "sainte": "ste", "marg": "mg", "nagar": "ngr", "sector": "sec",
    "main": "mn", "cross": "crs", "layout": "lyt", "colony": "col", "phase": "ph",
    "block": "blk", "industrial": "ind", "estate": "est", "area": "ar", "post": "po",
    "district": "dist", "taluk": "tq", "village": "vill", "junction": "jn",
    "bengaluru": "bangalore", "mumbai": "bombay", "chennai": "madras", "kolkata": "calcutta",
    "gurugram": "gurgaon", "puducherry": "pondicherry", "thiruvananthapuram": "trivandrum",
    # generic / US
    "expressway": "expy", "freeway": "fwy", "turnpike": "tpke", "crossing": "xing",
    "heights": "hts", "center": "ctr", "centre": "ctr", "point": "pt",
    "null": "", "none": "", "nan": "", "unknown": "",
    # India
    "flr": "fl", "grd": "gr", "ground": "gr", "apartments": "apts", "society": "soc",
    "bazar": "bzr", "bazaar": "bzr", "complex": "cmplx",
    # France (street types, articles and "bis" suffixes)
    "rue": "r", "bld": "blvd", "bvd": "blvd", "allees": "all", "chem": "ch", "cours": "crs",
    "esplanade": "espl", "passage": "pass", "residence": "res", "hameau": "ham",
    "bis": "", "de": "", "du": "", "des": "", "la": "", "le": "", "les": "", "l": "",
    "d": "", "aux": "",
}
LANDMARK_WORDS = {"nr", "opp", "bhd", "beside", "next", "landmark", "adjacent", "adj"}

_PUNCT = re.compile(r"[^a-z0-9 ]+")
_SPACE = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g",
                       "7": "t", "8": "b", "9": "g"})
_ORDINAL = re.compile(r"^\d+(st|nd|rd|th|e|er|eme|re)$")
_SINGLES = re.compile(r"\b(?:[a-z] )+[a-z]\b")


def _deleet(tok: str) -> str:
    """'hea1thcare' -> 'healthcare', 'c0rp' -> 'corp'; keeps '24hr', '1st', '3m'."""
    nd = sum(ch.isdigit() for ch in tok)
    if nd == 0 or nd > 2 or len(tok) - nd < 3 or _ORDINAL.match(tok):
        return tok
    return tok.translate(_LEET)


_WEB = re.compile(r"\b(?:https?://)?(?:www\.)?([a-z0-9-]+)\.(?:com|net|org|biz|info|co|in|us|fr|io)"
                  r"(?:\.[a-z]{2})?\b")
_ID = re.compile(r"\(?\b(?:id|ref|store|unit)\s*[:#]\s*\d+\)?|#\s*\d{4,}")


def basic(s, name=False) -> str:
    if not isinstance(s, str):
        return ""
    s = unidecode(s).lower()
    if name:
        # "ridgefresenius.com" -> "ridgefresenius"; drop "(ID: 78239)", "#36918" store ids
        s = _WEB.sub(r" \1 ", s.replace("www.", " "))
        s = _ID.sub(" ", s)
    s = s.replace("&", " and ").replace("@", " at ").replace("+", " and ")
    s = s.replace("'", "").replace("`", "")  # mcdonald's -> mcdonalds
    s = _PUNCT.sub(" ", s)
    if name:
        s = " ".join(_deleet(t) for t in s.split())
        # dotted acronyms / legal forms: "l l c" -> "llc", "p c" -> "pc", "i b m" -> "ibm"
        s = _SINGLES.sub(lambda m: m.group(0).replace(" ", ""), s)
    # split glued digits/letters: "12b" stays, "no12" -> "no 12"
    s = re.sub(r"([a-z]{2,})(\d)", r"\1 \2", s)
    return _SPACE.sub(" ", s).strip()


def _canon_tokens(s: str, table) -> list:
    out = []
    for t in s.split():
        t = table.get(t, t)
        if t:
            out.append(t)
    return out


# rough transliteration / phonetic skeleton so "Lakshmi"~"Laxmi", "Shree"~"Sri"
_SKEL_RULES = [
    (re.compile(r"ksh|x"), "ks"), (re.compile(r"(sh|ch|zh)"), "s"),
    (re.compile(r"ph"), "f"), (re.compile(r"(th|dh|bh|gh|kh|jh)"), lambda m: m.group(0)[0]),
    (re.compile(r"w"), "v"), (re.compile(r"z"), "s"), (re.compile(r"q"), "k"),
    (re.compile(r"c(?=[eiy])"), "s"), (re.compile(r"c"), "k"), (re.compile(r"ck"), "k"),
    (re.compile(r"y"), "i"), (re.compile(r"ee|ea|ie"), "i"), (re.compile(r"oo|ou"), "u"),
    (re.compile(r"aa"), "a"), (re.compile(r"([a-z])\1+"), r"\1"),
    (re.compile(r"(?<=[a-z])h\b"), ""),
]


def skeleton_token(t: str) -> str:
    if t.isdigit() or len(t) <= 2:
        return t
    for pat, rep in _SKEL_RULES:
        t = pat.sub(rep, t)
    # drop non-initial vowels: consonant skeleton keeps typo-robust identity
    return t[0] + re.sub(r"[aeiou]", "", t[1:]) if len(t) > 3 else t


_PRA_LI = re.compile(r"\bpraa? li(m)?\b")  # Indic abbreviation of "Pvt. Ltd."
_DBA = re.compile(r"\b(dba|doing business as|trading as)\b")


@lru_cache(maxsize=65536)
def norm_name(s) -> dict:
    b = basic(s, name=True)
    # DBA / trade-name: keep both parts, they are also tokens of the full string
    b = _DBA.sub(" dba ", b)
    b = _PRA_LI.sub("pvt ltd", b)
    toks = []
    for t in _canon_tokens(b, NAME_CANON):
        if len(t) >= 5 and t not in LEGAL and not t.isdigit():
            t = LEGAL_SKEL.get(skeleton_token(t), t)
        toks.append(t)
    core = [t for t in toks if t not in LEGAL] or toks
    skel = [skeleton_token(t) for t in core]
    acro = "".join(t[0] for t in core if not t.isdigit()) if len(core) >= 2 else ""
    return {
        "full": " ".join(toks),
        "core": " ".join(core),
        "skel": " ".join(skel),
        "sorted": " ".join(sorted(core)),
        "acro": acro,
        "first": core[0] if core else "",
    }


_POSTAL = re.compile(r"\b(\d{3}\s?\d{3}|\d{5}(?:\s?\d{4})?)\b")


@lru_cache(maxsize=65536)
def norm_addr(s) -> dict:
    b = basic(s)
    postals = set(p.replace(" ", "")[:6] for p in _POSTAL.findall(b))
    postals = {p for p in postals if len(p) >= 5}
    toks = []
    for t in _canon_tokens(b, ADDR_CANON):
        # house numbers: "0227" / "00572f" -> "227" / "572f" (postal codes untouched)
        if t[0] == "0" and t not in postals and (t.isdigit() or _DIGITS.fullmatch(t[:-1])):
            t = t.lstrip("0") or "0"
        toks.append(t)
    nums = [t for t in toks if t.isdigit() or _DIGITS.fullmatch(t[:-1] or "x")]
    nums = [n for n in nums if n.replace(" ", "")[:6] not in postals]
    words = [t for t in toks if not t[0].isdigit()]
    landmark = int(any(t in LANDMARK_WORDS for t in toks))
    return {
        "full": " ".join(toks),
        "words": " ".join(words),
        "nums": nums,
        "first_num": nums[0] if nums else "",
        "postal": postals,
        "landmark": landmark,
        "tail": " ".join(words[-2:]),  # usually city/state
        "skel": " ".join(skeleton_token(t) for t in words),
    }
