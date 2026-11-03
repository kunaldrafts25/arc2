"""Read-only observer for one accepted ARC2 official submission; never resubmits."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.submit_arc2_hybrid import COMPETITION


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',type=int,choices=[1,2],required=True)
    parser.add_argument('--submission-ref',type=int,required=True)
    args=parser.parse_args()
    directory=ROOT/f'reports/kaggle/arc2-inference-hybrid-v{args.version}'
    accepted=json.loads((directory/'official-submission.json').read_text())
    if accepted['phase']!='accepted' or accepted['submission_ref']!=args.submission_ref:
        raise ValueError('Observer must match the accepted ARC2 submission journal')
    token_path=ROOT/'.kaggle/access_token'
    os.environ['KAGGLE_API_TOKEN']=token_path.read_text(encoding='utf-8-sig').strip()
    os.environ['KAGGLE_CONFIG_DIR']=str(token_path.parent)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    until=datetime.now(timezone.utc)+timedelta(hours=18)
    while datetime.now(timezone.utc)<until:
        try:
            submission=api.get_submission(args.submission_ref)
            record=dict(submission_ref=args.submission_ref,status=submission.status.name,
                public_score=submission.public_score,error=submission.error_description,
                checked_at=datetime.now(timezone.utc).isoformat())
            temporary=directory/'official-status.tmp'
            temporary.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
            temporary.replace(directory/'official-status.json')
            with (directory/'official-history.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
            if record['status'] in ('COMPLETE','ERROR'):
                # This is an official score, unlike the public training metrics.
                print('ARC2_OFFICIAL_TERMINAL',json.dumps(record),flush=True)
                return
        except Exception as exc:
            print('Observation read failed:',type(exc).__name__,flush=True)
        time.sleep(60)
    print('Observer window expired; reconcile exact submission through read-only API.',flush=True)


if __name__=='__main__':
    main()
