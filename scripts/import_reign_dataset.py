import os
import json
import time
import requests
from pathlib import Path

import polars as pl

RAW_DATA_DIR = os.path.join("data", "raw")
DICTATORS_FILE = Path("data/dictators.txt")
CACHE_FILE = Path("data/wiki_cache.json")

AUTHORIARIAN_VALUES = [
    'party-personal',
    'personal',
    'party-military',
    'party-personal-military',
    'military',
    'warlordism',
    'military personal',
    'monarchy',
    'oligarchy',
    'party-based',
]
# According to Geddes, Wright & Frantz, which classifcation system has been used by the creators of
# REIGN dataset ('gwf_regimetype') (https://oefdatascience.github.io/REIGN.github.io/menu/reign_current.html)
# these types of regimes are considered authoritarian/anti-democratic/autocratic (https://datafinder.qog.gu.se/dataset/gwf)

NON_PERSON_KEYWORDS = [
    'list of', 'dictatorship', 'republic', 'revolution',
    'military', 'government', 'history of', 'politics of',
    'war', 'occupation', 'kingdom of', 'empire', 'coup',
    'party', 'election', 'santo', 'province', 'region',
    'genocide', 'massacre', 'battle of', 'crisis', 'conflict',
    'movement', 'administration', 'junta', 'committee',
]

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "UniversityRAGProject/1.1 (william.zehren@utc.fr)"
})


def load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def get_reign_dataset() -> pl.DataFrame:
    df_regime = pl.read_csv(
        os.path.join(RAW_DATA_DIR, "regime_list.csv"),
        encoding="utf8-lossy",
        infer_schema_length=10000,
    )
    df_leaders = pl.read_csv(
        os.path.join(RAW_DATA_DIR, "leader_list_8_21.csv"),
        encoding="utf8-lossy",
        infer_schema_length=10000,
    )

    df_leaders = df_leaders.with_columns(
        pl.concat_str(
            [pl.col('sdate'), pl.col('smonth'), pl.col('syear')],
            separator='/',
        ).alias('sfulldate')
    )

    df_leaders = df_leaders.with_columns(
        pl.concat_str(
            [pl.lit('1'), pl.col('emonth'), pl.col('eyear')],
            separator='/',
        ).alias('efulldate')
    )

    df_regime = df_regime.drop(['gwf_casename']).rename({'cowcode': 'ccode'})
    df_leaders = df_leaders.select(['ccode', 'leader', 'sfulldate', 'efulldate'])

    df_reign = df_leaders.join(df_regime, on='ccode')
    df_reign = df_reign.with_columns(
        pl.col('sfulldate').str.to_date(format="%d/%m/%Y").alias('sfulldate'),
        pl.col('efulldate').str.to_date(format="%d/%m/%Y").alias('efulldate'),
        pl.col('gwf_startdate').str.to_date(format="%m/%d/%Y").alias('gwf_startdate'),
        pl.col('gwf_enddate').str.to_date(format="%m/%d/%Y").alias('gwf_enddate'),
    )
    return df_reign


def get_authoritarian_leaders(reign_df: pl.DataFrame) -> pl.DataFrame:
    reign_df = reign_df.with_columns(
        pl.when(
            pl.col('sfulldate') >= pl.col('gwf_startdate'),
            pl.col('sfulldate') <= pl.col('gwf_enddate'),
            pl.col('efulldate') >= pl.col('gwf_startdate'),
            pl.col('efulldate') <= pl.col('gwf_enddate'),
        ).then(True).otherwise(False)
        .alias('corresponding_regime')
    )

    print(reign_df.select('gwf_regimetype','corresponding_regime', 'leader').filter(pl.col('leader')=="Aristide"))

    reign_df = reign_df.filter(pl.col('corresponding_regime'))
    reign_df = reign_df.filter(pl.col('gwf_regimetype').is_in(AUTHORIARIAN_VALUES))
    return reign_df


def is_likely_person(title: str) -> bool:
    """Retourne False si le titre ressemble à une page non-biographique."""
    if len(title.split()) < 2: 
        return False
    title_lower = title.lower()
    return not any(kw in title_lower for kw in NON_PERSON_KEYWORDS)


def is_human_on_wikidata(wikipedia_title: str) -> bool:
    """Vérifie via Wikidata que la page correspond à un être humain (Q5)."""
    try:
        params = {
            "action": "query",
            "titles": wikipedia_title,
            "prop": "pageprops",
            "format": "json",
        }
        response = SESSION.get("https://en.wikipedia.org/w/api.php", params=params, timeout=10)
        pages = response.json().get("query", {}).get("pages", {})
        for page in pages.values():
            wikidata_id = page.get("pageprops", {}).get("wikibase_item", "")
            if wikidata_id:
                wd_params = {
                    "action": "wbgetclaims",
                    "entity": wikidata_id,
                    "property": "P31",  # instanceof
                    "format": "json",
                }
                wd_resp = SESSION.get(
                    "https://www.wikidata.org/w/api.php", params=wd_params, timeout=10
                )
                claims = wd_resp.json().get("claims", {}).get("P31", [])
                for claim in claims:
                    val = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
                    if val.get("id") == "Q5":  # Q5 = humain
                        return True
    except Exception as e:
        print(f"  Erreur Wikidata pour '{wikipedia_title}': {e}")
    return False


def _search_wikipedia(query: str, search_type: str = "text", limit: int = 3) -> list[str] | None:
    """
    search_type='nearmatch' : cherche par titre uniquement → retourne [titre] ou []
    search_type='text'      : recherche plein texte → retourne une liste de titres
    """
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srwhat": search_type,
        "format": "json",
        "utf8": 1,
        "srlimit": limit,
    }
    for attempt in range(3):
        try:
            response = SESSION.get(
                "https://en.wikipedia.org/w/api.php", params=params, timeout=10
            )
            if response.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"\n  Rate limit atteint, attente {wait}s...")
                time.sleep(wait)
                continue
            if response.status_code != 200:
                return None
            results = response.json().get("query", {}).get("search", [])
            return [r["title"] for r in results]
        except Exception as e:
            print(f"\n  Erreur Wikipedia ({query}): {e}")
            return None
    return None


def _normalize_token(t: str) -> str:
    """Lowercase, strip ponctuation, supprime les accents pour la comparaison."""
    import unicodedata
    t = t.lower().strip("'.-")
    return unicodedata.normalize("NFD", t).encode("ascii", "ignore").decode()


def _is_better_than_raw(found: str, raw: str) -> bool:
    """
    Retourne False si le nom trouvé est moins informatif que le nom brut du dataset,
    ou si aucun token du nom original n'apparaît dans le résultat.
    Évite de remplacer "Lemus" par "Nayib Bukele" ou "Khalifah Ath-Thani" par "Na'im".
    - Le résultat doit contenir au moins autant de tokens que le nom brut.
    - Au moins un token du nom original doit correspondre exactement à un token du résultat
      (comparaison sans accents pour gérer "Mendez" == "Méndez").
    """
    if len(found.split()) < len(raw.split()):
        return False
    raw_tokens = {_normalize_token(t) for t in raw.split()}
    found_tokens = {_normalize_token(t) for t in found.split()}
    return bool(raw_tokens & found_tokens)  # intersection : token exact commun


def get_wikipedia_full_name(leader_name: str, country: str, cache: dict) -> str:
    """
    Cherche le nom complet Wikipedia d'un leader.
    1. Recherche par titre (nearmatch) → plus précis, évite les faux positifs plein texte
    2. Fallback recherche plein texte + validation Wikidata
    Utilise le cache pour éviter les requêtes redondantes.
    """
    cache_key = f"{leader_name}|{country}"
    if cache_key in cache:
        return cache[cache_key]

    title_results = _search_wikipedia(leader_name, search_type="nearmatch", limit=3)
    if title_results:
        for title in title_results:
            if is_likely_person(title) and _is_better_than_raw(title, leader_name):
                cache[cache_key] = title
                save_cache(cache)
                return title

    full_query = f"{leader_name} {country} politician"
    text_results = _search_wikipedia(full_query, search_type="text", limit=5)
    if text_results:
        for title in text_results:
            if is_likely_person(title) and is_human_on_wikidata(title) and _is_better_than_raw(title, leader_name):
                cache[cache_key] = title
                save_cache(cache)
                return title

    cache[cache_key] = leader_name
    save_cache(cache)
    return leader_name


def normalize_name(name: str) -> str:
    return name.strip().lower()


def compare_to_existing_data(reign_df: pl.DataFrame) -> None:
    """Compare le dataset avec dictators.txt et ajoute les nouveautés."""
    existing_normalized = set()
    if DICTATORS_FILE.exists():
        for line in DICTATORS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                existing_normalized.add(normalize_name(line))

    cache = load_cache()
    unique_leaders = reign_df.select(['leader', 'gwf_country']).unique().to_dicts()
    new_dictators = []

    print(f"Recherche Wikipedia pour {len(unique_leaders)} leaders uniques...")

    for i, row in enumerate(unique_leaders, 1):
        raw_name = row['leader']
        country = row['gwf_country']

        print(f"  [{i}/{len(unique_leaders)}] {raw_name} ({country})", end=" ... ")

        full_name = get_wikipedia_full_name(raw_name, country, cache)
        time.sleep(0.5)

        full_name_norm = normalize_name(full_name)

        if (normalize_name(raw_name) in existing_normalized
                or full_name_norm in existing_normalized):
            print("déjà présent")
            continue

        print(f"→ '{full_name}'")
        new_dictators.append(full_name)
        existing_normalized.add(full_name_norm)

    if new_dictators:
        print(f"\nAjout de {len(new_dictators)} nouveau(x) dictateur(s) dans {DICTATORS_FILE}...")
        DICTATORS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DICTATORS_FILE, mode='a', encoding='utf-8') as f:
            f.write("\n# Nouveaux depuis REIGN Dataset\n")
            for name in new_dictators:
                f.write(f"{name}\n")
                print(f"  + {name}")
    else:
        print("\nAucun nouveau dictateur trouvé, tout est à jour.")


if __name__ == "__main__":
    print("Analyse du dataset REIGN en cours...")

    df_reign = get_reign_dataset()

    df_auth = get_authoritarian_leaders(df_reign)
    print(f"{len(df_auth)} entrées autoritaires trouvées ({df_auth['leader'].n_unique()} leaders uniques)")

    compare_to_existing_data(df_auth)

    print("Terminé !")