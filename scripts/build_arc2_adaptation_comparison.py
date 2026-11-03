"""Matched before/after adaptation pilot on 50 fixed public TRAINING tasks."""
from hashlib import sha256
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def build():
    gate=ROOT/'reports/kaggle/arc2-adaptation-probe-v2'
    report=json.loads((gate/'adaptation-probe.json').read_text(encoding='utf-8'))
    if not report['passed'] or len(report['cases'])!=2:raise ValueError('Adaptation GPU probe has not passed')
    upload=json.loads((gate/'upload.json').read_text(encoding='utf-8'))
    for name,expected in upload['files'].items():
        if sha256((ROOT/'notebooks/arc2_adaptation_probe'/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Uploaded GPU probe source changed')
    files={name:(ROOT/name).read_text(encoding='utf-8') for name in
        ('arc2/__init__.py','arc2/symbolic.py','arc2/evaluation.py','arc2/neural.py','arc2/adaptation.py')}
    code='from pathlib import Path\nimport json,time,sys,subprocess\nSTART=time.monotonic()\nEND=START+43200-600\n'
    code+=f'sources={files!r}\n'
    code+='''for name,source in sources.items():
    path=Path('/kaggle/working')/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source)
sys.path.insert(0,'/kaggle/working')
subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=True,stdout=subprocess.DEVNULL)
from arc2.symbolic import solve
from arc2.neural import GridModel
from arc2.adaptation import TaskAdapter
from arc2.evaluation import score_submission
roots=[Path('/kaggle/input/datasets/kunalsinghh03/arc2-public-training-development'),Path('/kaggle/input/arc2-public-training-development')]
root=next(p for p in roots if p.is_dir())
models=[Path('/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1'),Path('/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1')]
backend=GridModel(next(p for p in models if (p/'config.json').is_file()));adapter=TaskAdapter(backend)
all_challenges=json.loads((root/'arc-agi_training_challenges.json').read_text())
challenges={k:all_challenges[k] for k in sorted(all_challenges)[:50]}
before={};after={};events=[]
def predict(task,fallback):
    result=[]
    for i,test in enumerate(task['test']):
        choices=[]
        for orientation in (0,4):
            prediction,info=backend.generate(task,i,orientation,deadline=min(END,time.monotonic()+120))
            if prediction is not None and prediction not in choices:choices.append(prediction)
        for prediction in fallback[i].values():
            if prediction not in choices:choices.append(prediction)
        result.append(dict(attempt_1=choices[0],attempt_2=choices[1] if len(choices)>1 else choices[0]))
    return result
for number,(key,task) in enumerate(challenges.items()):
    floor,_=solve(task,seconds=0.25)
    adapter.reset();before[key]=predict(task,floor)
    adaptation=adapter.adapt(task,steps=8,deadline=min(END,time.monotonic()+60))
    after[key]=predict(task,floor)
    events.append(dict(task_id=key,adaptation=adaptation))
    if (number+1)%5==0:
        Path('/kaggle/working/before-predictions.json').write_text(json.dumps(before))
        Path('/kaggle/working/after-predictions.json').write_text(json.dumps(after))
        print('PAIRED_ADAPTATION_PROGRESS',number+1,len(challenges),time.monotonic()-START,flush=True)
# Predictions are frozen before opening the public TRAINING solution labels.
Path('/kaggle/working/before-predictions.json').write_text(json.dumps(before))
Path('/kaggle/working/after-predictions.json').write_text(json.dumps(after))
all_targets=json.loads((root/'arc-agi_training_solutions.json').read_text());targets={k:all_targets[k] for k in challenges}
b=score_submission(challenges,targets,before);a=score_submission(challenges,targets,after)
improved=sum(not x['correct'] and y['correct'] for x,y in zip(b['per_output'],a['per_output']))
worsened=sum(x['correct'] and not y['correct'] for x,y in zip(b['per_output'],a['per_output']))
b.pop('per_output');a.pop('per_output')
report=dict(tasks=len(challenges),before=b,after=a,improved_outputs=improved,worsened_outputs=worsened,
    elapsed_seconds=time.monotonic()-START,events=events,
    boundary='Matched single deterministic pilot on 50 public training tasks. Pretraining may include these tasks; no unseen generalization, official improvement or statistical significance established.')
Path('/kaggle/working/adaptation-comparison.json').write_text(json.dumps(report,indent=2))
print('ARC2_ADAPTATION_COMPARISON_COMPLETE',json.dumps({k:v for k,v in report.items() if k!='events'}),flush=True)
'''
    compile(code,'arc2-adaptation-comparison','exec')
    metadata=json.loads((ROOT/'notebooks/arc2_adaptation_probe/kernel-metadata.json').read_text(encoding='utf-8'))
    metadata.update(id='kunalsinghh03/arc-hydra-arc2-adaptation-comparison',title='ARC Hydra ARC2 Adaptation Comparison')
    notebook=dict(nbformat=4,nbformat_minor=4,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3')),
        cells=[dict(cell_type='markdown',metadata={},source='# ARC2 matched task-local adaptation pilot\n\nFifty fixed public training tasks; same GPU/checkpoint, tokenization, grammar and identity/transpose predictions before and after eight demonstration-only LoRA steps. Reset adapters between tasks. Public training labels are read only after both prediction files are frozen. Not a competition submission or unseen evaluation. Maximum 12 hours including startup and a ten-minute finalization reserve.'),dict(cell_type='code',metadata={},source=code,outputs=[],execution_count=None)])
    output=ROOT/'notebooks/arc2_adaptation_comparison';output.mkdir(exist_ok=True)
    for name,value in [('submission.ipynb',notebook),('kernel-metadata.json',metadata)]:
        (output/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(kernel=metadata['id'],source_sha256=sha256(code.encode()).hexdigest(),tasks=50)))


if __name__=='__main__':build()
