"""Build the DEXPI class/property vocabulary used to validate the DEXPI export.

Source: the official machine-readable DEXPI models published with the specification
(https://dexpi.org/specification/2.0.1/html/, files `_static/Core.xml` and
`_static/Plant.xml`, CC BY 4.0). Each model lists its packages and classes
(`ConcreteClass` / `AbstractClass`, `superTypes`) with their composition, reference
and data properties; templates are already expanded into properties.

  python scripts/build_dexpi_vocab.py [--version 2.0.1] [--src DIR]

Without --src the model files are downloaded. Output:
src/pid/export/dexpi_xsd/dexpi_vocabulary.json
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from lxml import etree

from pid.config import resolve

SPEC = "https://dexpi.org/specification/{version}/html/_static/{model}.xml"
MODELS = ("Core", "Plant")
CLASS_TAGS = {"ConcreteClass": False, "AbstractClass": True}
PROPERTY_TAGS = {"CompositionProperty", "ReferenceProperty", "DataProperty"}


def load_model(model: str, version: str, src: Path | None) -> etree._Element:
    if src is not None:
        data = (src / f"{model}.xml").read_bytes()
    else:
        url = SPEC.format(version=version, model=model)
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
    root = etree.fromstring(data)
    if root.get("name") != model:
        raise ValueError(f"{model}.xml declares model {root.get('name')!r}")
    return root


def qualify(ref: str, model: str) -> str:
    """'/Piping.PipeFitting' (same model) -> 'Plant/Piping.PipeFitting'."""
    return f"{model}{ref}" if ref.startswith("/") else ref


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="2.0.1")
    ap.add_argument("--src", help="directory with Core.xml / Plant.xml (else download)")
    args = ap.parse_args()
    src = Path(args.src) if args.src else None

    classes: dict[str, dict[str, object]] = {}
    for model in MODELS:
        root = load_model(model, args.version, src)
        for el in root.iter(*CLASS_TAGS):
            pkg = el.getparent()
            pkg_name = pkg.get("name") if pkg is not None and pkg.tag == "Package" else None
            name = f"{model}/{pkg_name + '.' if pkg_name else ''}{el.get('name')}"
            sups = [qualify(s, model) for s in (el.get("superTypes") or "").split()]
            props = sorted(p.get("name") for p in el if p.tag in PROPERTY_TAGS)
            classes[name] = {"abstract": CLASS_TAGS[el.tag], "supertypes": sups,
                             "properties": props}
    dest = resolve("src/pid/export/dexpi_xsd/dexpi_vocabulary.json")
    dest.write_text(json.dumps({
        "dexpi_version": args.version,
        "source": SPEC.format(version=args.version, model="{Core,Plant}") + " (CC BY 4.0)",
        "classes": dict(sorted(classes.items()))}, indent=1))
    n_conc = sum(not v["abstract"] for v in classes.values())
    print(f"{len(classes)} classes ({n_conc} concrete) -> {dest}")


if __name__ == "__main__":
    main()
