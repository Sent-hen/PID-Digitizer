# GraphML output schema (version 1.0)

The pipeline's primary output is one GraphML file per drawing (`src/pid/export/graphml.py`).

- **Compatibility with the dataset:** it uses the attribute names and label spellings of the PID2Graph ground truth, so prediction files and ground-truth files are read by the same loader.
- **Readers:** any GraphML reader works, including NetworkX `read_graphml`, yEd, Gephi and igraph.
- **Validation:** every file validates against the official GraphML 1.0 XSD, vendored in `src/pid/export/xsd/`.

## Structure

```xml
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="..." for="graph|node|edge" attr.name="..." attr.type="..."/>   <!-- table below -->
  <graph id="<drawing id>" edgedefault="undirected">
    <data key="g_schema">1.0</data> ...                                    <!-- graph metadata -->
    <node id="s12"> <data key="label">valve</data> <data key="xmin">...</data> ... </node>
    <node id="background0"> <data key="label">background</data> ... </node> <!-- ignore region -->
    <edge id="e7" source="s12" target="a40"> <data key="edge_label">solid</data> ... </edge>
  </graph>
</graphml>
```

## Attributes

| element | `attr.name` | type | meaning |
|---|---|---|---|
| graph | `schema_version` | string | `1.0`. Readers reject a different major version. |
| graph | `drawing_id`, `subset` | string | drawing identity |
| graph | `source_file` | string | input image file name |
| graph | `width`, `height` | int | input image size in pixels |
| graph | `pipeline_version` | string | git commit of the code that produced the file |
| graph | `config_hash` | string | hash of the pipeline + detector configuration |
| node | `label` | string | node class (below) |
| node | `xmin`, `ymin`, `xmax`, `ymax` | double | bounding box in input-image pixels (x right, y down) |
| node | `confidence` | double | detector score for symbols. For structural nodes, the mean confidence of their edges. |
| edge | `edge_label` | string | `solid` (process line) or `non-solid` (dashed/dotted signal line) |
| edge | `confidence` | double | calibrated probability that the edge is correct (RESULTS R-008) |

## Node labels

Labels use the raw PID2Graph spellings.

| `label` | meaning |
|---|---|
| `general`, `tank`, `valve`, `instrumentation`, `pump`, `arrow`, `inlet/outlet` | symbols |
| `crossing` | 8×8 structural node where lines meet: a bend (degree 2), T (3) or X (4). Crossing lines are always connected, as in PID2Graph. |
| `connector` | 8×8 structural node where a line enters a symbol (an "ankle", i.e. a port), on the symbol's bounding-box border |
| `background` | ignore region (frame or title block), degree 0. Not a graph node. |

Internally, `connector` is called `ankle` and `inlet/outlet` is called `inlet_outlet` (`configs/data.yaml`).

## Guarantees and limits

- **Exact round trip:** `Drawing → GraphML → Drawing` is exact, and `GraphML → Drawing → GraphML` is byte-identical (floats written with `repr`; edge order and endpoint orientation preserved). Tested in `tests/test_graphml.py`.
- **IDs:** node ids are XML NMTOKENs. PID2Graph OPEN100 ids such as `inlet/outlet3` are sanitized to `inlet_outlet3` on load.
- **Not encoded:** line geometry and edge polylines (edges run between node centres), text and tags (not produced), and ports beyond ankle nodes.
- **Versioning:** changing these attributes after Milestone 4 requires a schema version bump and user approval (CLAUDE.md).
