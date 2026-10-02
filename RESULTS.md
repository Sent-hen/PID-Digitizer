# Results

Every number reported anywhere in this repo must appear here, with the config, commit, hardware and command that produced it. Raw records are appended to `outputs/eval_log.jsonl` by `scripts/evaluate.py`.

Hardware: see DECISIONS.md D-001.

## R-000 Harness oracle (ground truth vs itself)

Sanity check, not a result. Command: `python scripts/evaluate.py --oracle --split dataset_pid_test --split synthetic_test --split open100_test`. Config `configs/data.yaml`, commit `70a6dc5` + Milestone 0 working tree.

| split | drawings | Symbols mAP@0.5 | mAP@0.5:0.95 | Nodes AP@0.5 | Edge P / R / F1 | Edge mAP (PID2Graph) |
|---|---|---|---|---|---|---|
| dataset_pid_test | 75 | 1.0 | 1.0 | 1.0 | 1.0 / 1.0 / 1.0 | 1.0 |
| synthetic_test | 38 | 1.0 | 1.0 | 1.0 | 1.0 / 1.0 / 1.0 | 1.0 |
| open100_test | 12 | 1.0 | 1.0 | 1.0 | 1.0 / 1.0 / 1.0 | 1.0 |

Perturbation tests (`tests/test_oracle.py`) confirm that deleting k edges, adding spurious edges, dropping nodes, shifting boxes below IoU 0.5, flipping line types and relabelling classes change each metric by exactly the expected amount.

## Baselines

None yet (Milestone 1).
