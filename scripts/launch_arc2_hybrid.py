"""One journaled, free-quota-only launch of the original-account ARC2 candidate."""
from datetime import datetime, timezone
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',type=int,choices=[1,2],default=1)
    args=parser.parse_args()
    notebook = ROOT/('notebooks/arc2_inference_hybrid' if args.version==1 else 'notebooks/arc2_inference_hybrid_v2')
    manifest = json.loads((notebook/'manifest.json').read_text())
    assert manifest['kernel'] == 'kunalsinghh25/arc-hydra-arc2-inference-hybrid'
    for name, digest in manifest['files'].items():
        if sha256((notebook/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Candidate changed: '+name)
    token_path = ROOT/'.kaggle/access_token'
    os.environ['KAGGLE_API_TOKEN'] = token_path.read_text(encoding='utf-8-sig').strip()
    os.environ['KAGGLE_CONFIG_DIR'] = str(token_path.parent)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    quota = api.quota_view().gpu_quota
    available = (quota.total_time_allowed-quota.time_used-quota.time_reserved).total_seconds()
    required=900 if args.version==1 else 650
    if quota.is_pay_to_scale_enabled or available < required:
        raise RuntimeError(f'Need at least {required} unreserved free GPU seconds, with paid scaling disabled')
    if args.version==2 and api.kernels_status(manifest['kernel']+'/1').status.name!='COMPLETE':
        raise RuntimeError('Version 1 must be complete before targeted version 2')
    directory = ROOT/f'reports/kaggle/arc2-inference-hybrid-v{args.version}'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'launch.json'
    # Exclusive journal is written before the non-idempotent mutation. If the
    # call is interrupted or uncertain, reconcile through read-only API calls.
    state = dict(phase='launching', checked_at=datetime.now(timezone.utc).isoformat(),
                 available_gpu_seconds=available, **manifest)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(state, stream, indent=2)
    try:
        response = api.kernels_push(str(notebook))
        state.update(phase='accepted', response=response.to_dict())
    except Exception as exc:
        state.update(phase='outcome_unconfirmed', error_type=type(exc).__name__)
    path.write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
