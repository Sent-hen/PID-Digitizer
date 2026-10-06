DEXPI 2.0.1 artefacts used to validate the DEXPI-structured export. Both come from the
DEXPI Specification 2.0.1 (released 2026-09-30), https://dexpi.org/specification/2.0.1/html/.
© DEXPI Initiative, licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

- `DEXPI_XML_Schema.xsd`: the official DEXPI XML Schema, unmodified
  (`_static/DEXPI_XML_Schema.xsd` of the specification site; identical to release V2.0.1 of
  https://gitlab.com/dexpi/Specification).
- `dexpi_vocabulary.json`: classes (abstract or concrete), supertypes and properties of the
  official Core and Plant models (`_static/Core.xml`, `_static/Plant.xml`), extracted by
  `scripts/build_dexpi_vocab.py`.

Checks:
- The specification's reference P&ID (`_static/reference_pid.xml`) passes `validate_dexpi`
  with zero errors.
- The 2.0.1 changelog items that affect Plant are present, e.g. `Instrumentation.ProcessSafetyFunction`
  and `ProcessEquipment.Heater.TubeBundle`.
