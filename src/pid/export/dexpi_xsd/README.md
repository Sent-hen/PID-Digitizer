DEXPI 2.0.1 artefacts used to validate the DEXPI-structured export.

- `DEXPI_XML_Schema.xsd`: the official DEXPI XML Schema, copied unmodified from
  https://gitlab.com/dexpi/Specification (`src/documentation/_static/`, release V2.0.1,
  2026-09-30). © DEXPI Initiative, licensed CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/).
- `dexpi_vocabulary.json`: classes (abstract or concrete), supertypes and properties, derived
  by `scripts/build_dexpi_vocab.py` from the DEXPI model sources in the same repository
  (`src/model/**/*.py`). It is an adaptation of the DEXPI model, CC BY 4.0, © DEXPI Initiative.

Validated against: the DEXPI 2.0.1 reference P&ID (`reference_pid.xml` in the same
repository) passes `validate_dexpi` with zero errors.
