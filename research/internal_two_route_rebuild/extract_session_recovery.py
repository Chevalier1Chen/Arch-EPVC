import json
from pathlib import Path

src = Path(r"C:\Users\DELL\.codex\sessions\2026\07\31\rollout-2026-07-31T11-58-57-019fb653-43ca-7a30-9493-f0d87bd84060.jsonl")
out = Path(r"C:\Users\DELL\Documents\论文\internal_two_route_rebuild\session_recovery_hits.txt")
needles = [
    "hourly_architecture_comparison", "best_hourly_lstm", "shared_scalers",
    "best_multimodal_tabtransformer", "best_tabular_mlp", "best_tabular_resnet",
    "platform_model_package", "HourlyMultimodalModel", "hourly_scalers",
]

hits = []
with src.open("r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        if not any(n.lower() in line.lower() for n in needles):
            continue
        try:
            obj = json.loads(line)
            pretty = json.dumps(obj, ensure_ascii=False, indent=2)
        except Exception:
            pretty = line
        hits.append(f"\n{'=' * 30} LINE {line_no} {'=' * 30}\n{pretty[:30000]}\n")

out.write_text("".join(hits), encoding="utf-8")
print(f"hits={len(hits)}")
print(out)
