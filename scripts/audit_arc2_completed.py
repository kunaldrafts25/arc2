"""Independently rescore frozen ARC2 development outputs on permitted training labels."""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arc2.evaluation import score_submission


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    data = ROOT/'reference/arc2/data'
    challenges = read(data/'arc-agi_training_challenges.json')
    targets = read(data/'arc-agi_training_solutions.json')
    raw = ROOT/'reference/kaggle/arc2-neural-development-v1'
    paired = ROOT/'reference/kaggle/arc2-adaptation-comparison-v1'
    reported = read(raw/'neural-development-metrics.json')
    scores = {}
    flags = {}
    hashes = {}
    for name in ('neural', 'symbolic', 'hybrid'):
        path = raw/(name+'-predictions.json')
        result = score_submission(challenges, targets, read(path))
        flags[name] = {(r['task_id'], r['test_index']): r['correct'] for r in result.pop('per_output')}
        if result != reported['scores'][name]:
            raise ValueError('Downloaded predictions do not reproduce reported '+name+' score')
        scores[name] = result
        hashes[path.name] = sha256(path.read_bytes()).hexdigest()
    events = read(raw/'neural-events.json')
    attempts = Counter(e.get('reason', 'valid_output') for e in events)
    outputs_with_valid = {(e['task_id'], e['test_index']) for e in events if not e.get('reason')}
    expected_outputs = sum(len(t['test']) for t in challenges.values())
    # The exact source package must still match the record from before upload.
    for slug, notebook in [('arc2-neural-development-v1', 'arc2_neural_development'),
                           ('arc2-adaptation-comparison-v1', 'arc2_adaptation_comparison')]:
        launch = read(ROOT/'reports/kaggle'/slug/'launch.json')
        for name, digest in launch['files'].items():
            if sha256((ROOT/'notebooks'/notebook/name).read_bytes()).hexdigest() != digest:
                raise ValueError('Uploaded source package changed: '+slug+'/'+name)
    subset = {key: challenges[key] for key in sorted(challenges)[:50]}
    labels = {key: targets[key] for key in subset}
    paired_report = read(paired/'adaptation-comparison.json')
    paired_scores = {}
    paired_flags = {}
    for name in ('before', 'after'):
        path = paired/(name+'-predictions.json')
        result = score_submission(subset, labels, read(path))
        paired_flags[name] = {(r['task_id'], r['test_index']): r['correct'] for r in result.pop('per_output')}
        if result != paired_report[name]:
            raise ValueError('Adaptation score does not reproduce: '+name)
        paired_scores[name] = result
        hashes[name+'-predictions.json'] = sha256(path.read_bytes()).hexdigest()
    gained = [dict(task_id=k[0], test_index=k[1]) for k in paired_flags['before']
              if not paired_flags['before'][k] and paired_flags['after'][k]]
    lost = [dict(task_id=k[0], test_index=k[1]) for k in paired_flags['before']
            if paired_flags['before'][k] and not paired_flags['after'][k]]
    if len(gained) != paired_report['improved_outputs'] or len(lost) != paired_report['worsened_outputs']:
        raise ValueError('Paired improvements/regressions do not reproduce')
    hybrid_gained = sum(not flags['neural'][k] and flags['hybrid'][k] for k in flags['neural'])
    hybrid_lost = sum(flags['neural'][k] and not flags['hybrid'][k] for k in flags['neural'])
    training = [e['adaptation'] for e in paired_report['events']]
    report = dict(checked_at=datetime.now(timezone.utc).isoformat(),
                  neural_kernel='kunalsinghh03/arc-hydra-arc2-neural-development/1',
                  paired_kernel='kunalsinghh03/arc-hydra-arc2-adaptation-comparison/1',
                  scores_reproduced=True, scores=scores,
                  hybrid_gained_outputs=hybrid_gained, hybrid_lost_outputs=hybrid_lost,
                  generation_outcomes=dict(attempts), expected_test_outputs=expected_outputs,
                  outputs_with_valid_neural_generation=len(outputs_with_valid),
                  outputs_without_valid_neural_generation=expected_outputs-len(outputs_with_valid),
                  neural_elapsed_seconds=reported['elapsed_seconds'],
                  paired_scores=paired_scores, adaptation_gained=gained, adaptation_lost=lost,
                  adaptation_steps_completed=sum(t['steps_completed'] for t in training),
                  adaptation_stop_reasons=dict(Counter(t['stop_reason'] for t in training)),
                  artifact_hashes=hashes,
                  decision='Preserve inference-only hybrid as the stronger measured development candidate. Do not promote this eight-step LoRA setting; it regressed in the fixed paired pilot.',
                  boundary='Both datasets are public TRAINING tasks and may have appeared in checkpoint pretraining. Neural baseline includes symbolic fallback. The adaptation pilot has a different 50-task subset; its percentages cannot be compared directly with the 1000-task benchmark. No held-out or official improvement established; official ARC2 best remains 0.00.')
    destination = ROOT/'reports/arc2/completed-development-v1'
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'audit.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    shutil.copy2(raw/'neural-development-metrics.json', ROOT/'reports/kaggle/arc2-neural-development-v1/neural-development-metrics.json')
    shutil.copy2(paired/'adaptation-comparison.json', ROOT/'reports/kaggle/arc2-adaptation-comparison-v1/adaptation-comparison.json')
    print(json.dumps({k: v for k, v in report.items() if k != 'artifact_hashes'}, indent=2))


if __name__ == '__main__':
    main()
