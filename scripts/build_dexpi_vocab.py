"""Build the DEXPI 2.0 class/property vocabulary used to validate our DEXPI export.

The DEXPI model is published as Python-DSL source files in the official repository
(https://gitlab.com/dexpi/Specification, CC BY 4.0). This script parses those files
with `ast` (never executes them) and writes, for every class, whether it is abstract,
its supertypes and its own properties:

  python scripts/build_dexpi_vocab.py <dir-with-model-.py-files> [--version 2.0.1]

Output: src/pid/export/dexpi_xsd/dexpi_vocabulary.json
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from pid.config import resolve

CLASS_CALLS = {"CONCRETE_CLASS": False, "ABSTRACT_CLASS": True}
PROPERTY_CALLS = {"COMPOSITION_PROPERTY", "REFERENCE_PROPERTY", "DATA_PROPERTY"}


def chain(node: ast.AST) -> list[str] | None:
    """`A.B.C` -> ['A', 'B', 'C']"""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return parts[::-1]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="directory containing the DEXPI model .py files")
    ap.add_argument("--version", default="2.0.1")
    args = ap.parse_args()

    models: set[str] = set()
    alias: dict[str, tuple[str, str | None]] = {}  # var -> (model, package)
    classes: dict[str, dict[str, object]] = {}
    props: dict[str, set[str]] = {}
    pending_super: dict[str, list[list[str]]] = {}

    def resolve_name(parts: list[str]) -> str | None:
        """['PIPING', 'PipeTee'] -> 'Plant/Piping.PipeTee'; ['Core', 'X'] -> 'Core/X'."""
        head, rest = parts[0], parts[1:]
        if head in alias:
            model, pkg = alias[head]
        elif head in models:
            model, pkg = head, None
        else:
            return None
        if pkg is None and len(rest) >= 2:  # Plant.Piping.PipeTee
            pkg, rest = rest[0], rest[1:]
        if len(rest) != 1:
            return None
        return f"{model}/{pkg + '.' if pkg else ''}{rest[0]}"

    files = sorted(Path(args.src).glob("*.py"))
    # Two passes: aliases/models first (files reference each other's packages).
    trees = [ast.parse(f.read_text(encoding="utf-8")) for f in files]
    for tree in trees:
        for st in tree.body:
            if not isinstance(st, ast.Assign):
                continue
            val = st.value
            call = val.func.id if isinstance(val, ast.Call) and isinstance(val.func, ast.Name) \
                else None
            for tgt in st.targets:
                parts = chain(tgt)
                if not parts:
                    continue
                if call == "MODEL" and len(parts) == 1:
                    models.add(parts[0])
                    alias[parts[0]] = (parts[0], None)
            if call in ("PACKAGE", "MODEL"):
                for tgt in st.targets:
                    parts = chain(tgt)
                    if parts and len(parts) == 2 and parts[0] in models:
                        for t2 in st.targets:
                            p2 = chain(t2)
                            if p2 and len(p2) == 1:
                                alias[p2[0]] = (parts[0], parts[1])
            elif call is None and isinstance(val, ast.Attribute):
                src = chain(val)
                if src and len(src) == 2 and src[0] in models:
                    for tgt in st.targets:
                        p = chain(tgt)
                        if p and len(p) == 1:
                            alias[p[0]] = (src[0], src[1])
    for tree in trees:
        for st in tree.body:
            if not (isinstance(st, ast.Assign) and isinstance(st.value, ast.Call)
                    and isinstance(st.value.func, ast.Name)):
                continue
            fn = st.value.func.id
            for tgt in st.targets:
                parts = chain(tgt)
                if not parts:
                    continue
                if fn in CLASS_CALLS:
                    name = resolve_name(parts)
                    if name is None:
                        continue
                    sup: list[list[str]] = []
                    tmpl: set[str] = set()
                    for kw in st.value.keywords:
                        if kw.arg == "superTypes" and isinstance(kw.value, ast.List):
                            sup = [c for c in (chain(e) for e in kw.value.elts) if c]
                        # templates=[PLANT_TEMPLATES.Impellers, ...] add a property
                        # named after the template to the class
                        if kw.arg == "templates" and isinstance(kw.value, ast.List):
                            tmpl = {c[-1] for c in (chain(e) for e in kw.value.elts) if c}
                    classes[name] = {"abstract": CLASS_CALLS[fn]}
                    pending_super[name] = sup
                    props.setdefault(name, set()).update(tmpl)
                elif fn in PROPERTY_CALLS and len(parts) >= 3:
                    owner = resolve_name(parts[:-1])
                    if owner is not None:
                        props.setdefault(owner, set()).add(parts[-1])
    out = {}
    for name, info in sorted(classes.items()):
        sups = [s for s in (resolve_name(p) for p in pending_super[name]) if s]
        out[name] = {"abstract": info["abstract"], "supertypes": sups,
                     "properties": sorted(props.get(name, set()))}
    dest = resolve("src/pid/export/dexpi_xsd/dexpi_vocabulary.json")
    dest.write_text(json.dumps({"dexpi_version": args.version,
                                "source": "https://gitlab.com/dexpi/Specification (CC BY 4.0)",
                                "classes": out}, indent=1))
    print(f"{len(out)} classes ({sum(not v['abstract'] for v in out.values())} concrete) "
          f"-> {dest}")


if __name__ == "__main__":
    main()
