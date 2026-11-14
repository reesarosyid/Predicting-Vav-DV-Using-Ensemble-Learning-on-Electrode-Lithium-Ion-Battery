import argparse
import gc
import json
import os
import time
from collections import defaultdict
from fractions import Fraction
from functools import reduce
from math import gcd

import pandas as pd
import requests
from mp_api.client import MPRester
from pymatgen.core import Composition, Element

import config

AFLUX_BASE = "http://aflow.org/API/aflux/"
AFLUX_KEYS = ("compound,species,composition,enthalpy_cell,"
              "enthalpy_formation_cell,"
              "volume_cell,spacegroup_relax,natoms,nspecies")

CRYSTAL_SYSTEMS = (
    (2, "Triclinic"),
    (15, "Monoclinic"),
    (74, "Orthorhombic"),
    (142, "Tetragonal"),
    (167, "Trigonal"),
    (194, "Hexagonal"),
    (230, "Cubic"),
)


def crystal_system_from_sg(sg_number):
    """Crystal system for a space group number (1-230), IUCr ranges.

    AFLOW only returns the space group number, but the crystal system is
    used as a feature.
    """
    if not isinstance(sg_number, int) or not 1 <= sg_number <= 230:
        return None
    for upper, name in CRYSTAL_SYSTEMS:
        if sg_number <= upper:
            return name
    return None


def source_of(mat):
    """'MP' or 'AFLOW'. Falls back to the id prefix for older files."""
    if mat.get("source"):
        return mat["source"]
    return "MP" if str(mat.get("id", "")).startswith("mp-") else "AFLOW"


def composition_from_dict(comp):
    return Composition("".join(f"{el}{amt}" for el, amt in comp.items()))


def load_json(path):
    if not os.path.exists(path):
        raise SystemExit(f"Input file not found: {path}\n"
                         "Run the previous step first (see --help).")
    with open(path) as f:
        return json.load(f)


def save_json(data, path, indent=2):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ---------------------------------------------------------------------------
# Step 1: fetch materials
# ---------------------------------------------------------------------------
def _mp_composition(val):
    """Arrow composition column -> {element: int count}."""
    if val is None:
        return None
    if isinstance(val, dict):
        items = val.items()
    elif isinstance(val, str):
        items = json.loads(val).items()
    else:
        try:
            items = [(k, v) for k, v in val]
        except (TypeError, ValueError):
            return None
    out = {str(k): int(round(v)) for k, v in items if v is not None}
    return out or None


def _mp_symmetry(val, key):
    if val is None:
        return None
    if isinstance(val, dict):
        return val.get(key)
    return getattr(val, key, None)


def fetch_mp_materials(api_key, output_path=config.RAW_MP_PATH, limit=None):
    """Fetch MP materials from the local Arrow copy of the summary dataset.

    Energies are taken from `formation_energy_per_atom` (MP2020 corrected).
    The raw `uncorrected_energy_per_atom` is not comparable across entries
    because MP mixes PAW/functional settings.
    """
    fields = ["material_id", "formula_pretty", "composition", "nsites",
              "volume", "energy_per_atom", "formation_energy_per_atom",
              "symmetry"]

    with MPRester(api_key) as mpr:
        print("\nLoading MP summary dataset...")
        dataset = getattr(mpr.materials.summary.search(), "pyarrow_dataset", None)
        if dataset is None:
            raise RuntimeError("mp-api client has no .pyarrow_dataset; "
                               "upgrade mp-api.")
        columns = [c for c in fields if c in dataset.schema.names]
        df = dataset.to_table(columns=columns).to_pandas()

    if limit:
        df = df.head(limit)
    print(f"Processing {len(df):,} rows...")

    materials = []
    for row in df.to_dict("records"):
        ef = row.get("formation_energy_per_atom")
        vol = row.get("volume")
        # skip non-solids: no formation energy or no valid volume
        if ef is None or pd.isna(ef) or vol is None or pd.isna(vol) or vol <= 0:
            continue
        comp = _mp_composition(row.get("composition"))
        if not comp:
            continue

        e = row.get("energy_per_atom")
        sg = _mp_symmetry(row.get("symmetry"), "number")
        cs = _mp_symmetry(row.get("symmetry"), "crystal_system")
        nsites = int(row["nsites"])
        materials.append({
            "id": str(row["material_id"]),
            "source": "MP",
            "formula": row.get("formula_pretty"),
            "composition": comp,
            "nsites": nsites,
            "volume": float(vol),
            "formation_energy_cell": float(ef * nsites),
            "energy_cell": None if e is None or pd.isna(e) else float(e * nsites),
            "space_group": None if sg is None else int(sg),
            "crystal_system": None if cs is None else str(cs),
        })

    save_json(materials, output_path)
    print(f"\nFetched {len(materials):,} materials -> {output_path}")
    return materials


def _query_aflux(directive, n_per_page, page, retries=3):
    parts = [AFLUX_KEYS]
    if directive:
        parts.append(directive)
    parts.append(f"$paging({page},{n_per_page})")
    url = f"{AFLUX_BASE}?{','.join(parts)}"

    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if attempt == retries - 1:
                print(f"[AFLOW] page {page} failed: {e}")
                return []
            time.sleep(5 * (attempt + 1))
    return []


def _parse_aflow_entry(d):
    comp_str = d.get("compound")
    if not comp_str:
        return None
    volume = float(d.get("volume_cell") or 0)
    energy = d.get("enthalpy_cell")
    ef = d.get("enthalpy_formation_cell")
    if ef is None or energy is None or volume <= 0:
        return None

    comp = Composition(comp_str)
    comp_dict = {str(el): int(round(amt))
                 for el, amt in comp.get_el_amt_dict().items()}
    sg = d.get("spacegroup_relax")
    sg_number = int(sg) if isinstance(sg, (int, float)) else None
    return {
        "id": d.get("auid") or comp_str,
        "source": "AFLOW",
        "formula": comp.reduced_formula,
        "composition": comp_dict,
        "nsites": int(d.get("natoms", sum(comp_dict.values()))),
        "volume": volume,
        "formation_energy_cell": float(ef),
        "energy_cell": float(energy),
        "space_group": sg_number,
        "crystal_system": crystal_system_from_sg(sg_number),
    }


def fetch_aflow_materials(out_path=config.RAW_AFLOW_PATH, n_per_page=1000,
                          max_pages=1000, restrict_to_active_metals=True):
    """Fetch AFLOW materials page by page.

    restrict_to_active_metals=True queries `species(*M*)` per active metal.
    It is much lighter, but misses charged end members that no longer
    contain the metal (e.g. CoO2 for LiCoO2). False pulls the whole AFLOW
    repository.
    """
    if restrict_to_active_metals:
        tasks = [(m, f"species(*{m}*)") for m in config.ACTIVE_METALS]
    else:
        tasks = [("ALL", "")]

    records, seen = [], set()
    for label, directive in tasks:
        print(f"Fetching AFLOW materials [{label}]...")
        for page in range(1, max_pages + 1):
            batch = _query_aflux(directive, n_per_page, page)
            if not batch:
                break

            for d in batch:
                try:
                    rec = _parse_aflow_entry(d)
                except Exception:
                    continue
                if rec is None:
                    continue
                key = (rec["formula"], rec["space_group"])
                if key not in seen:
                    seen.add(key)
                    records.append(rec)

            print(f"  {label} page {page}: {len(batch)} records, "
                  f"total: {len(records)}")
            if len(batch) < n_per_page:
                break
            time.sleep(1)

    save_json(records, out_path)
    print(f"\nFetched {len(records):,} AFLOW materials -> {out_path}")
    return records


def combine_materials(mp_path, aflow_path, output_path):
    """Merge MP and AFLOW, deduplicating on (source, formula, space group).

    Duplicates are only removed within a source, since pairs are never
    matched across sources.
    """
    mp_materials = load_json(mp_path)
    aflow_materials = load_json(aflow_path)
    print(f"MP materials: {len(mp_materials):,}")
    print(f"AFLOW materials: {len(aflow_materials):,}")

    seen, combined = set(), []
    for mat in mp_materials + aflow_materials:
        key = (source_of(mat), mat["formula"], mat["space_group"])
        if key not in seen:
            seen.add(key)
            combined.append(mat)

    n_mp = sum(source_of(m) == "MP" for m in combined)
    print(f"Combined: {len(combined):,} materials "
          f"({n_mp:,} MP + {len(combined) - n_mp:,} AFLOW)")
    save_json(combined, output_path)
    return combined


def _aflow_element_references(aflow_path=config.RAW_AFLOW_PATH):
    """Lowest energy per atom of each pure active metal in the AFLOW data."""
    try:
        with open(aflow_path) as f:
            mats = json.load(f)
    except FileNotFoundError:
        print(f"  [AFLOW] {aflow_path} not found, skipping")
        return {}

    e_metal = {}
    for metal in config.ACTIVE_METALS:
        energies = [m["energy_cell"] / sum(m["composition"].values())
                    for m in mats
                    if set(m.get("composition") or {}) == {metal}
                    and sum(m["composition"].values()) > 0]
        if energies:
            e_metal[metal] = float(min(energies))
            print(f"  [AFLOW] {metal}: {e_metal[metal]:.4f} eV/atom")
        else:
            print(f"  [AFLOW] {metal}: no pure phase")
    return e_metal


def fetch_element_reference_energies(api_key,
                                     output_path=config.ELEMENT_REF_PATH):
    """Pure-metal reference energies E_M per source (for reference only).

    V_av is computed from formation energies, where E_M cancels out. The
    sources are kept apart because their energy scales differ (pure Li is
    -1.90 eV/atom in AFLOW vs -2.39 eV/atom in MP).
    """
    print("\nFetching element reference energies (E_M)...")
    e_mp = {}
    with MPRester(api_key) as mpr:
        for metal in config.ACTIVE_METALS:
            try:
                docs = mpr.materials.summary.search(
                    chemsys=metal,
                    fields=["material_id", "energy_per_atom", "energy_above_hull"],
                )
            except Exception as e:
                print(f"  [MP] {metal}: error {e}")
                continue
            if not docs:
                print(f"  [MP] {metal}: no pure phase")
                continue
            best = min(docs, key=lambda d: d.energy_above_hull
                       if d.energy_above_hull is not None else float("inf"))
            if best.energy_per_atom is None:
                print(f"  [MP] {metal}: missing energy_per_atom")
                continue
            e_mp[metal] = float(best.energy_per_atom)
            print(f"  [MP] {metal}: {e_mp[metal]:.4f} eV/atom ({best.material_id})")

    e_metal = {"MP": e_mp, "AFLOW": _aflow_element_references()}
    save_json(e_metal, output_path)
    print(f"Saved E_M to: {output_path}")
    return e_metal


def validate_energy_consistency(path=config.COMBINED_RAW, threshold=5.0):
    """Check that formation energies are comparable within each source.

    For a given formula, the per-atom formation energy should only differ
    by polymorph-scale amounts. A spread above `threshold` eV/atom for more
    than 1% of formulas means the energies (and therefore V_av) cannot be
    trusted. Returns True if every source passes.
    """
    print_header("VALIDATION: formation energy consistency per source")
    mats = load_json(path)

    per_source = defaultdict(lambda: defaultdict(list))
    n_missing = 0
    for m in mats:
        ef = m.get("formation_energy_cell")
        n = sum((m.get("composition") or {}).values())
        if ef is None:
            n_missing += 1
        elif n:
            per_source[source_of(m)][m["formula"]].append(ef / n)

    if n_missing:
        print(f"  {n_missing:,} materials have no formation_energy_cell and "
              "will be skipped. Re-run step 1.")

    ok = bool(per_source)
    for src, groups in sorted(per_source.items()):
        spreads = sorted(max(v) - min(v) for v in groups.values() if len(v) > 1)
        if not spreads:
            print(f"  [{src}] no formula with more than one entry, skipped")
            continue
        n = len(spreads)
        frac_bad = sum(s > threshold for s in spreads) / n
        passed = frac_bad < 0.01
        ok &= passed
        print(f"  [{src}] {n:,} formulas | median {spreads[n // 2]:.3f} eV | "
              f"worst {spreads[-1]:.1f} eV | {100 * frac_bad:.1f}% above "
              f"{threshold} eV -> {'OK' if passed else 'FAIL'}")

    print("\n  " + ("Formation energies are consistent."
                    if ok else
                    "Formation energies are NOT consistent; V_av is unreliable."))
    return ok


# ---------------------------------------------------------------------------
# Step 2: charged/discharged pairs
# ---------------------------------------------------------------------------
def _framework_fingerprint(comp, metal):
    """Reduced composition of the non-metal framework, or None if empty."""
    framework = {el: n for el, n in comp.items() if el != metal}
    if not framework:
        return None
    g = reduce(gcd, framework.values())
    if g == 0:
        return None
    return tuple(sorted((el, n // g) for el, n in framework.items()))


def _same_framework_elements(comp_a, comp_b, metal):
    return set(comp_a) - {metal} == set(comp_b) - {metal}


def _commensurate_factor(comp_a, comp_b, metal):
    """Scale factor f with comp_a = f * comp_b on every framework element.

    Returns (f, ok); ok is False if the frameworks are not commensurate.
    """
    framework = (set(comp_a) | set(comp_b)) - {metal}
    ref = next((el for el in sorted(framework)
                if comp_a.get(el, 0) > 0 and comp_b.get(el, 0) > 0), None)
    if ref is None:
        return None, False

    f = Fraction(comp_a[ref], comp_b[ref])
    ok = all(Fraction(comp_b.get(el, 0)) * f == comp_a.get(el, 0)
             for el in framework)
    return f, ok


def _scaled_side(mat, metal, factor):
    """One electrode of a pair, scaled to the common framework size."""
    energy = mat["energy_cell"]
    return {
        "id": mat["id"],
        "formula": mat["formula"],
        "comp": mat["composition"],
        "E": energy * factor if energy is not None else None,
        "Ef": mat["formation_energy_cell"] * factor,
        "V": mat["volume"] * factor,
        "xM": mat["composition"].get(metal, 0) * factor,
        "sg": mat["space_group"],
        "cs": mat["crystal_system"],
    }


def find_pairs(materials, metals=None):
    """Find all valid charged/discharged electrode pairs.

    Two materials form a pair when their non-metal frameworks are
    commensurate and their metal content differs. Pairs are only built
    within one source, since energies from MP and AFLOW are not comparable.
    """
    metals = metals or config.ACTIVE_METALS
    pairs = []

    for metal in metals:
        buckets = defaultdict(list)
        for mat in materials:
            fp = _framework_fingerprint(mat["composition"], metal)
            if fp is not None:
                buckets[(source_of(mat), fp)].append(mat)

        for (src, _), group in buckets.items():
            for i, a in enumerate(group):
                for b in group[i + 1:]:
                    comp_a, comp_b = a["composition"], b["composition"]
                    if not _same_framework_elements(comp_a, comp_b, metal):
                        continue
                    f, ok = _commensurate_factor(comp_a, comp_b, metal)
                    if not ok:
                        continue
                    if (a.get("formation_energy_cell") is None
                            or b.get("formation_energy_cell") is None):
                        continue

                    # scale the smaller cell up to the larger one
                    if f > 1:
                        side_a = _scaled_side(a, metal, 1.0)
                        side_b = _scaled_side(b, metal, float(f))
                    else:
                        side_a = _scaled_side(a, metal, float(1 / f))
                        side_b = _scaled_side(b, metal, 1.0)

                    if abs(side_a["xM"] - side_b["xM"]) < config.RATIO_TOL:
                        continue

                    # the side with more metal is the discharged one
                    disch, chg = ((side_a, side_b) if side_a["xM"] > side_b["xM"]
                                  else (side_b, side_a))
                    pairs.append({
                        "metal": metal,
                        "valence": config.VALENCE[metal],
                        "source": src,
                        "discharged": disch,
                        "charged": chg,
                        "x1": chg["xM"],
                        "x2": disch["xM"],
                    })
    return pairs


# ---------------------------------------------------------------------------
# Step 3: electrode properties (Eq. 1-4)
# ---------------------------------------------------------------------------
def _ion_properties(metal):
    """First and second ionization energies (eV) of the working ion."""
    try:
        ies = Element(metal).ionization_energies
    except Exception:
        return None, None
    first = ies[0] if len(ies) > 0 else None
    second = ies[1] if len(ies) > 1 else None
    return first, second


def compute_properties(pair):
    """V_av, dV%, specific capacity and specific energy of one pair.

    Returns None if the pair is invalid.
    """
    metal, q = pair["metal"], pair["valence"]
    disch, chg = pair["discharged"], pair["charged"]
    x1, x2 = pair["x1"], pair["x2"]
    dx = x2 - x1
    if dx <= 0 or disch.get("Ef") is None or chg.get("Ef") is None:
        return None

    # Eq. 1 written with formation energies. Both sides share the same
    # framework, so the framework terms and E_M cancel out:
    #   V_av = -(Ef_disch - Ef_chg) / (q * dx)
    v_av = -(disch["Ef"] - chg["Ef"]) / (q * dx)

    # Eq. 2: volume change
    denom = min(disch["V"], chg["V"])
    if denom <= 0:
        return None
    delta_v = abs(disch["V"] - chg["V"]) / denom * 100.0

    # Eq. 3: specific capacity (mAh/g), molar mass on the same scaled basis as dx
    metal_count = disch["comp"].get(metal, 0)
    if metal_count <= 0:
        return None
    molar_mass = composition_from_dict(disch["comp"]).weight * (x2 / metal_count)
    if molar_mass <= 0:
        return None
    capacity = q * dx * config.FARADAY * 1000.0 / (3600.0 * molar_mass)

    # Eq. 4: specific energy (Wh/kg)
    specific_energy = capacity * v_av

    ion_first, ion_second = _ion_properties(metal)

    return {
        "metal": metal,
        "valence": q,
        "source": pair.get("source") or source_of(chg),
        "discharged_id": disch["id"],
        "discharged_formula": disch["formula"],
        "charged_id": chg["id"],
        "charged_formula": chg["formula"],
        "discharged_sg": disch["sg"],
        "charged_sg": chg["sg"],
        "discharged_cs": disch["cs"],
        "charged_cs": chg["cs"],
        "x1_charged": x1,
        "x2_discharged": x2,
        "ion_property_first": ion_first,
        "ion_property_second": ion_second,
        "V_av": round(v_av, 4),
        "dV_percent": round(delta_v, 4),
        "capacity_mAh_g": round(capacity, 4),
        "specific_energy_Wh_kg": round(float(specific_energy), 4),
    }


# ---------------------------------------------------------------------------
# Step 4: composition features (matminer)
# ---------------------------------------------------------------------------
def build_featurizer():
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.composition import (
        ElementProperty, IonProperty, Stoichiometry, ValenceOrbital,
    )

    featurizer = MultipleFeaturizer([
        Stoichiometry(),
        ElementProperty.from_preset("magpie"),
        ValenceOrbital(props=["frac"]),
        IonProperty(fast=True),
    ])
    # n_jobs=1 avoids the multiprocessing pool, which uses far more RAM
    featurizer.set_n_jobs(1)
    return featurizer, featurizer.feature_labels()


def featurize_batch(featurizer, labels, pairs, props):
    """Properties plus matminer features for both electrodes (disch__/chg__)."""
    df = pd.DataFrame(props)
    for side, prefix in (("discharged", "disch__"), ("charged", "chg__")):
        df["_comp"] = [composition_from_dict(p[side]["comp"]) for p in pairs]
        df = featurizer.featurize_dataframe(df, "_comp", ignore_errors=True,
                                            pbar=False)
        df = df.rename(columns={c: prefix + c for c in labels})
        df = df.drop(columns="_comp")
    return df


class BatchWriter:
    """Append DataFrames to a CSV or Parquet file, chosen by extension."""

    def __init__(self, path):
        self.path = path
        self.is_parquet = str(path).endswith(".parquet")
        self.columns = None
        self._parquet_writer = None
        self._write_header = True
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(path):
            os.remove(path)

    def write(self, df):
        # keep the column order of the first batch so the schema stays fixed
        if self.columns is None:
            self.columns = list(df.columns)
        df = df.reindex(columns=self.columns)

        if self.is_parquet:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(df, preserve_index=False)
            if self._parquet_writer is None:
                self._parquet_writer = pq.ParquetWriter(self.path, table.schema)
            self._parquet_writer.write_table(table)
        else:
            df.to_csv(self.path, mode="a", header=self._write_header, index=False)
            self._write_header = False

    def close(self):
        if self._parquet_writer is not None:
            self._parquet_writer.close()


def stream_pairs(pairs_path, batch_size=2000, max_pairs=None,
                 properties_out=None, features_out=None):
    """Read the pairs file incrementally and write properties and/or features.

    The pairs file can be several GB, so it is parsed with ijson and
    processed in batches instead of being loaded at once.
    """
    try:
        import ijson
    except ImportError:
        raise SystemExit("ijson is not installed: pip install ijson")

    if properties_out is None and features_out is None:
        raise ValueError("stream_pairs needs at least one output path")

    featurizer = labels = None
    if features_out:
        featurizer, labels = build_featurizer()
    prop_writer = BatchWriter(properties_out) if properties_out else None
    feat_writer = BatchWriter(features_out) if features_out else None

    batch_pairs, batch_props = [], []
    n_read = n_valid = 0

    def flush():
        nonlocal n_valid
        if not batch_props:
            return
        if prop_writer:
            prop_writer.write(pd.DataFrame(batch_props))
        if feat_writer:
            feat_writer.write(
                featurize_batch(featurizer, labels, batch_pairs, batch_props))
        n_valid += len(batch_props)
        batch_pairs.clear()
        batch_props.clear()
        gc.collect()

    outputs = ", ".join(p for p in (properties_out, features_out) if p)
    print(f"\nStreaming {pairs_path} (batch={batch_size}) -> {outputs}")
    with open(pairs_path, "rb") as f:
        for pair in ijson.items(f, "item", use_float=True):
            n_read += 1
            props = compute_properties(pair)
            if props is not None:
                batch_pairs.append(pair)
                batch_props.append(props)
            if len(batch_pairs) >= batch_size:
                flush()
            if n_read % 200_000 == 0:
                print(f"  ...{n_read:,} read | {n_valid:,} valid", flush=True)
            if max_pairs and n_read >= max_pairs:
                break
    flush()

    if prop_writer:
        prop_writer.close()
        print(f"  properties -> {properties_out}")
    if feat_writer:
        feat_writer.close()
        print(f"  features   -> {features_out}")
    print(f"Done: {n_read:,} pairs read, {n_valid:,} valid.")
    return n_valid


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def step1_fetch(api_key, limit=None):
    print_header("STEP 1: FETCH MATERIALS (MP + AFLOW)")
    fetch_mp_materials(api_key, limit=limit)
    fetch_aflow_materials()
    combine_materials(config.RAW_MP_PATH, config.RAW_AFLOW_PATH, config.COMBINED_RAW)
    fetch_element_reference_energies(api_key)
    validate_energy_consistency()


def step2_find_pairs():
    print_header("STEP 2: FIND CHARGED/DISCHARGED PAIRS")
    pairs = find_pairs(load_json(config.COMBINED_RAW))
    save_json(pairs, config.PAIRS_PATH, indent=None)
    print(f"Found {len(pairs):,} pairs -> {config.PAIRS_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Battery electrode data pipeline: fetch, pair, compute "
                    "properties and featurize.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Steps: 1=fetch (MP+AFLOW)  2=find pairs  3=properties  4=features\n"
            "Steps 2-4 run offline. Only step 1 needs MP_API_KEY.\n\n"
            "Examples:\n"
            "  python scraping.py\n"
            "  python scraping.py --steps 2,3,4\n"
            "  python scraping.py --steps 4 --features-out Dataset/07_features.parquet\n"
            "  python scraping.py --steps 4 --max-pairs 5000"))
    parser.add_argument("--steps", default="all",
                        help="'all' or a comma-separated list of 1,2,3,4")
    parser.add_argument("--batch-size", type=int, default=2000,
                        help="streaming batch size (default: 2000)")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="stop after this many pairs in steps 3/4")
    parser.add_argument("--limit", type=int, default=None,
                        help="limit the number of MP rows in step 1")
    parser.add_argument("--features-out", default=config.FEATURES_PATH,
                        help="feature output for step 4, .csv or .parquet "
                             f"(default: {config.FEATURES_PATH})")
    parser.add_argument("--validate", action="store_true",
                        help="only run the energy consistency check and exit")
    args = parser.parse_args()

    if args.validate:
        raise SystemExit(0 if validate_energy_consistency() else 1)

    if args.steps.strip().lower() == "all":
        steps = {"1", "2", "3", "4"}
    else:
        steps = {s.strip() for s in args.steps.split(",") if s.strip()}
    unknown = steps - {"1", "2", "3", "4"}
    if unknown:
        raise SystemExit(f"Unknown steps: {sorted(unknown)}. Use 1,2,3,4 or 'all'.")

    print(f"\nActive metals: {', '.join(config.ACTIVE_METALS)}")
    print(f"Valence: {config.VALENCE}")
    print(f"Running steps: {', '.join(sorted(steps))}")

    if "1" in steps:
        api_key = os.environ.get("MP_API_KEY")
        if not api_key:
            raise SystemExit("MP_API_KEY is not set (required for step 1).")
        step1_fetch(api_key, limit=args.limit)
    if "2" in steps:
        step2_find_pairs()

    # steps 3 and 4 share a single pass over the pairs file
    if "3" in steps or "4" in steps:
        print_header("STEP 3/4: PROPERTIES AND FEATURES")
        stream_pairs(
            config.PAIRS_PATH, args.batch_size, args.max_pairs,
            properties_out=config.PROPERTIES_PATH if "3" in steps else None,
            features_out=args.features_out if "4" in steps else None,
        )


if __name__ == "__main__":
    main()
