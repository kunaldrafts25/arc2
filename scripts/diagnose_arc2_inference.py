"""Diagnose frozen inference events without fitting a policy to target answers."""
from collections import Counter
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    events = json.loads((ROOT/'reference/kaggle/arc2-neural-development-v1/neural-events.json').read_text())
    grouped = {}
    for event in events:
        grouped.setdefault(event.get('reason', 'valid'), []).append(event)
    summary = {}
    for reason, rows in grouped.items():
        lengths = sorted(r['prompt_tokens'] for r in rows if 'prompt_tokens' in r)
        durations = sorted(r['elapsed_seconds'] for r in rows if 'elapsed_seconds' in r)
        summary[reason] = dict(count=len(rows),
            prompt_tokens=dict(min=min(lengths), median=statistics.median(lengths), max=max(lengths)) if lengths else None,
            elapsed_seconds=dict(total=sum(durations), median=statistics.median(durations), max=max(durations)) if durations else None,
            generated_token_counts=dict(Counter(r.get('generated_tokens') for r in rows)))
    report = dict(outcomes=summary,
        oom_examples=[r for r in events if r.get('reason') == 'GPU memory exhausted'][:4],
        first_budget_skip=next((r for r in events if r.get('reason') == 'budget'), None),
        findings=[
            'The old task-major loop spends both views on earlier tasks before later tasks get a first view.',
            'The 960-token cap already exceeds the maximum valid 30x30 grid plus EOS (930 tokens). Unfinished outputs are time limits, not a too-small token cap.',
            'OOM prompt sizes motivate chunked prefill with an unchanged KV cache and grammar; actual memory reduction and numerical parity require GPU measurements.',
            'Use view-major scheduling, durable complete symbolic fallback, per-output deadlines and a finalization reserve. No target labels used to choose these fixes.'])
    output = ROOT/'reports/arc2/inference-diagnostics-v1'
    output.mkdir(parents=True, exist_ok=True)
    (output/'diagnosis.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
