# Example: one drawing in, graph out

These are real outputs of the service (`docker compose up` → upload → download), committed so you can see what the system produces without running it.

| file | what it is |
|---|---|
| [`../../docker/sample/dataset_pid_10.png`](../../docker/sample/dataset_pid_10.png) | input drawing, 7168 × 4561 px |
| [`input_preview.jpg`](input_preview.jpg) | the input, downscaled for viewing |
| [`overlay_confidence.jpg`](overlay_confidence.jpg) | detected graph drawn over the input, colored by confidence (red → green) |
| [`result.graphml`](result.graphml) | the graph: 410 nodes, 458 edges ([schema](../../docs/GRAPHML_SCHEMA.md)) |
| [`result.dexpi.xml`](result.dexpi.xml) | the same graph as DEXPI 2.0.1 XML (DEXPI-structured; [coverage](../../docs/DEXPI_COVERAGE.md)) |

| Input | Output (confidence overlay) |
|---|---|
| ![input](input_preview.jpg) | ![overlay](overlay_confidence.jpg) |

## Provenance

- **Drawing:** sheet 10 of the Dataset-P&ID subset of PID2Graph. It's a validation sheet, never used for training or for the reported test results.
- **Produced by:** the CPU Docker image with detector `detector_m2c.pt` (sha1 `531ab26ce150`) and the final pipeline configuration.
- **Run time:** 146 s on CPU, about 5 s on an RTX 4070 Laptop GPU.
- **Validity:** both files validate (GraphML 1.0 XSD; DEXPI XSD + DEXPI 2.0.1 vocabulary), and the DEXPI file is byte-identical to `to_dexpi()` of the GraphML.
- **Reproduce:** run `make demo`, which uploads this sheet. Download the files from the URLs it prints.

## How good is it on this sheet?

Scored against the PID2Graph ground truth (406 nodes, 457 edges) with the project's evaluator:

| Symbols mAP@0.5 | Nodes AP@0.5 | Edge precision / recall / F1 | Edge mAP (PID2Graph protocol) |
|---|---|---|---|
| 1.000 | 0.934 | 0.908 / 0.910 / 0.909 | 0.913 |

That's typical for this subset: the val average edge F1 is 0.917 (RESULTS R-008). Most misses are junction or port nodes placed 2–4 px from the annotation ([error analysis](../../docs/ERROR_ANALYSIS.md)). Real scanned drawings are much harder (OPEN100 test edge F1 0.49); see the main README's Limitations.

## What the files contain

**GraphML.** A valve and one of its connections. Ankle nodes (`a…`) are where a pipe enters a symbol; `n_conf` and `e_conf` are confidences.

```xml
<node id="s38">
  <data key="label">valve</data>
  <data key="xmin">3109.613</data> <data key="ymin">3046.195</data>
  <data key="xmax">3171.995</data> <data key="ymax">3109.545</data>
  <data key="n_conf">0.9999</data>
</node>
<edge id="e149" source="a189" target="s38">
  <data key="edge_label">solid</data>
  <data key="e_conf">0.9603</data>
</edge>
```

**DEXPI.** A piping segment from a pipe tee to a valve. Junctions become `PipeTee`s, valves become `OperatedValve`s, and each port is a `PipingNode`.

```xml
<Object type="Plant/Piping.PipingNetworkSegment" id="Seg_ch0">
  <References property="SourceItem" objects="#Tee_c2"/>
  <References property="SourceNode" objects="#Node_c2_ch0a"/>
  <Components property="Items">
    <Object type="Plant/Piping.PipeTee" id="Tee_c2"> … 3 PipingNodes … </Object>
    <Object type="Plant/Piping.OperatedValve" id="Vl_s57"> … 2 PipingNodes … </Object>
  </Components>
  <References property="TargetItem" objects="#Vl_s57"/>
  <References property="TargetNode" objects="#Node_a109"/>
  <Components property="Connections">
    <Object type="Plant/Piping.Pipe" id="Pipe_ch0"> … </Object>
  </Components>
</Object>
```

**What's in the DEXPI file:**
- 49 valves and 16 instruments;
- 75 pipe tees and 197 piping segments in 7 piping systems;
- 31 instrument-to-piping links.

**Not in the DEXPI file:** 38 `general` symbols and 3 arrows, which have no DEXPI class.

## License

The drawing and these derived outputs are CC BY-SA 4.0.
- PID2Graph: Stürmer, Graumann & Koch (2025).
- Dataset-P&ID: Paliwal et al. (2021).
