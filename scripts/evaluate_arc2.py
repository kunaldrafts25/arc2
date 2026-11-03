"""Development evaluation on permitted public TRAINING tasks only.

Labels are read only after all predictions have been frozen on disk.
This is not an unseen test or an official competition result.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from arc2.symbolic import solve
from arc2.evaluation import score_submission


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds-per-task',type=float,default=0.25)
    args=parser.parse_args()
    source=ROOT/'reference/arc2/data';output=ROOT/'reports/arc2/symbolic-v1';output.mkdir(parents=True,exist_ok=True)
    challenges=json.loads((source/'arc-agi_training_challenges.json').read_text(encoding='utf-8'))
    started=time.monotonic();predictions={};events=[]
    for i,(key,task) in enumerate(sorted(challenges.items())):
        guesses,info=solve(task,seconds=args.seconds_per_task);predictions[key]=guesses
        events.append(dict(task_id=key,**info))
        if (i+1)%100==0:print('Predicted',i+1,'training tasks',flush=True)
    data=json.dumps(predictions,sort_keys=True).encode()
    (output/'predictions.json').write_bytes(data)
    solutions=json.loads((source/'arc-agi_training_solutions.json').read_text(encoding='utf-8'))
    score=score_submission(challenges,solutions,predictions)
    (output/'per-output.json').write_text(json.dumps(score.pop('per_output'),indent=2)+'\n',encoding='utf-8')
    report=dict(**score,tasks=len(challenges),elapsed_seconds=time.monotonic()-started,
        seconds_per_task=args.seconds_per_task,predictions_sha256=sha256(data).hexdigest(),
        tasks_with_consistent_program=sum(e.get('consistent_programs',0)>0 for e in events),
        timed_out_tasks=sum(e.get('timed_out',False) for e in events),
        boundary='Public training-task development measurement only. No held-out generalization or official score established.')
    (output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (output/'search-events.json').write_text(json.dumps(events,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
