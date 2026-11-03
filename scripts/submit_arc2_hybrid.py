"""Validate the completed preview and submit its exact version once, or inspect it."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arc2.evaluation import validate_submission

COMPETITION = 'arc-prize-2026-arc-agi-2'
KERNEL = 'kunalsinghh25/arc-hydra-arc2-inference-hybrid'
DIRECTORY = ROOT/'reports/kaggle/arc2-inference-hybrid-v1'
VERSION = 1


def validate_preview():
    launch = json.loads((DIRECTORY/'launch.json').read_text())
    if launch['phase'] != 'accepted' or launch['response']['versionNumber'] != VERSION:
        raise ValueError('Exact accepted version required')
    for name, digest in launch['files'].items():
        notebook=ROOT/('notebooks/arc2_inference_hybrid' if VERSION==1 else 'notebooks/arc2_inference_hybrid_v2')
        if sha256((notebook/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen source changed: '+name)
    raw = ROOT/f'reference/kaggle/arc2-inference-hybrid-v{VERSION}'
    report = json.loads((raw/'competition-run.json').read_text())
    if report['rerun'] or report['decoder'] != 'chunked' or report['valid_generations'] < 1:
        raise ValueError('Expected bounded preview with actual valid chunked generations')
    if len(report['parity_checks']) != 2 or not all(c['equal'] for c in report['parity_checks']):
        raise ValueError('GPU parity gate failed')
    memory = report['memory_check']
    if not memory or memory['info'].get('reason') == 'GPU memory exhausted':
        raise ValueError('Long-context memory check missing or failed')
    if VERSION==2 and not memory['valid']:
        raise ValueError('Version 2 must complete the long-context diagnostic')
    sample = json.loads((ROOT/'reference/arc2/data/sample_submission.json').read_text())
    challenges = {k:dict(test=[dict(input=[[0]]) for _ in rows]) for k, rows in sample.items()}
    submission = json.loads((raw/'submission.json').read_text())
    validate_submission(challenges, submission)
    if report['tasks'] != len(challenges) or report['test_outputs'] != sum(len(t['test']) for t in challenges.values()):
        raise ValueError('Preview coverage differs from competition sample')
    evidence = dict(validated_at=datetime.now(timezone.utc).isoformat(), tasks=report['tasks'],
        test_outputs=report['test_outputs'], decoder=report['decoder'], valid_generations=report['valid_generations'],
        source_files=launch['files'], output_files={p.name:sha256(p.read_bytes()).hexdigest()
            for p in raw.glob('*.json')},
        boundary='Functional GPU parity, long-context memory and format validation; no accuracy established. The long-context example may still hit its decode time limit.')
    (DIRECTORY/'validation.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
    (DIRECTORY/'competition-run.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return evidence


def inspect(api, ref):
    submission = api.get_submission(ref)
    record = dict(submission_ref=ref, status=submission.status.name,
                  public_score=submission.public_score, error=submission.error_description,
                  checked_at=datetime.now(timezone.utc).isoformat())
    (DIRECTORY/'official-status.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2))


def main():
    global VERSION, DIRECTORY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect', type=int)
    parser.add_argument('--version',type=int,choices=[1,2],default=1)
    args = parser.parse_args()
    VERSION=args.version
    DIRECTORY=ROOT/f'reports/kaggle/arc2-inference-hybrid-v{VERSION}'
    token_path = ROOT/'.kaggle/access_token'
    os.environ['KAGGLE_API_TOKEN'] = token_path.read_text(encoding='utf-8-sig').strip()
    os.environ['KAGGLE_CONFIG_DIR'] = str(token_path.parent)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    if args.inspect:
        return inspect(api, args.inspect)
    evidence = validate_preview()
    if api.kernels_status(KERNEL+'/'+str(VERSION)).status.name != 'COMPLETE':
        raise RuntimeError('Exact candidate version is not complete')
    quota = api.quota_view()
    if quota.gpu_quota.is_pay_to_scale_enabled:
        raise RuntimeError('Paid scaling must be disabled')
    limits = api.competition_get_submission_limits(COMPETITION)
    if limits.num_allowed_now < 1:
        raise RuntimeError('No current ARC2 submission allowance')
    submissions = api.competition_submissions(COMPETITION)
    if any(s.status.name not in ('COMPLETE','ERROR') for s in submissions):
        raise RuntimeError('Another ARC2 official evaluation is active')
    state = dict(phase='submitting', kernel=KERNEL+'/'+str(VERSION), competition=COMPETITION,
        checked_at=datetime.now(timezone.utc).isoformat(), known_refs=[s.ref for s in submissions],
        gpu_used_seconds=quota.gpu_quota.time_used.total_seconds(),
        gpu_reserved_seconds=quota.gpu_quota.time_reserved.total_seconds(),
        allowance=limits.num_allowed_now, validation=evidence)
    path = DIRECTORY/'official-submission.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(state, stream, indent=2)
    try:
        response = api.competition_submit_code(file_name='submission.json', competition=COMPETITION,
            kernel=KERNEL, kernel_version=VERSION, quiet=True,
            message='ARC2 inference-only hybrid: symbolic consistency, identity/transpose, chunked prefill parity gate, coverage-first scheduling. Official accuracy unverified.')
        state.update(phase='accepted' if int(response.ref or 0)>0 else 'outcome_unconfirmed',
                     submission_ref=int(response.ref or 0))
    except Exception as exc:
        state.update(phase='outcome_unconfirmed', error_type=type(exc).__name__)
    path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(state,indent=2))


if __name__ == '__main__':
    main()
