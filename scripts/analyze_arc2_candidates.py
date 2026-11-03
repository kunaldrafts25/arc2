"""Frozen-candidate development audit; no GPU and no new weight training."""
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arc2.evaluation import score_submission


def shape(value):
    return len(value), len(value[0])


def plausible_shapes(task, index):
    """Union of simple size rules consistent with every supplied demonstration."""
    pairs = task['train']
    if not pairs:
        return set()
    x = shape(task['test'][index]['input'])
    expected = set()
    outputs = {shape(p['output']) for p in pairs}
    if len(outputs) == 1:
        expected.update(outputs)
    for transpose in (False, True):
        for numerator in range(1, 6):
            for denominator in range(1, 6):
                def transform(dimensions):
                    h, w = dimensions[::-1] if transpose else dimensions
                    if h*numerator % denominator or w*numerator % denominator:
                        return None
                    return h*numerator//denominator, w*numerator//denominator
                if all(transform(shape(p['input'])) == shape(p['output']) for p in pairs):
                    transformed = transform(x)
                    if transformed and all(1 <= d <= 30 for d in transformed):
                        expected.add(transformed)
    return expected


def main():
    raw = ROOT/'reference/kaggle/arc2-neural-development-v1'
    challenges = json.loads((ROOT/'reference/arc2/data/arc-agi_training_challenges.json').read_text())
    predictions = {name:json.loads((raw/(name+'-predictions.json')).read_text())
                   for name in ('neural','symbolic','hybrid')}
    candidate_union = {}
    filtered = {}
    for key, task in sorted(challenges.items()):
        filtered[key] = []
        for index in range(len(task['test'])):
            choices = []
            for name in ('hybrid','neural','symbolic'):
                for guess in predictions[name][key][index].values():
                    if guess not in choices:
                        choices.append(guess)
            candidate_union[key,index] = choices
            expected = plausible_shapes(task, index)
            ordered = sorted(choices, key=lambda g: bool(expected) and shape(g) not in expected)
            filtered[key].append(dict(attempt_1=ordered[0], attempt_2=ordered[min(1,len(ordered)-1)]))
    destination = ROOT/'reports/arc2/candidate-analysis-v1'
    destination.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(filtered).encode()
    # Predictions frozen before opening any public training test target labels.
    frozen = ROOT/'reference/arc2/shape-filter-development-predictions.json'
    frozen.write_bytes(encoded)
    labels = json.loads((ROOT/'reference/arc2/data/arc-agi_training_solutions.json').read_text())
    baseline = score_submission(challenges, labels, predictions['hybrid'])
    alternate = score_submission(challenges, labels, filtered)
    gains = []
    losses = []
    for a,b in zip(baseline['per_output'],alternate['per_output']):
        if not a['correct'] and b['correct']:
            gains.append(dict(task_id=a['task_id'],test_index=a['test_index']))
        if a['correct'] and not b['correct']:
            losses.append(dict(task_id=a['task_id'],test_index=a['test_index']))
    # This is a diagnostic upper bound, not an implementable selector or score.
    oracle = sum(labels[key][i] in guesses for (key,i),guesses in candidate_union.items())
    baseline.pop('per_output'); alternate.pop('per_output')
    report = dict(baseline=baseline, shape_filter=alternate, gained=gains, lost=losses,
                  frozen_predictions_sha256=sha256(encoded).hexdigest(),
                  union_oracle_correct_outputs=oracle, union_oracle_total_outputs=len(candidate_union),
                  boundary='Public training-only analysis with a potentially contaminated checkpoint. Oracle uses target labels solely as a diagnostic upper bound. Shape selector uses demonstrations only. No held-out or official improvement; this policy is not in the submitted candidate.')
    (destination/'analysis.json').write_text(json.dumps(report,indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
