"""Build a standalone CPU ARC-2 floor and a separate tiny GPU capability probe."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def build(gpu_probe=False):
    metadata=dict(id='kunalsinghh25/arc-hydra-arc2-symbolic-baseline',title='ARC Hydra ARC2 Symbolic Baseline',
        code_file='submission.ipynb',language='python',kernel_type='notebook',is_private=True,
        enable_gpu=False,enable_tpu=False,enable_internet=False,dataset_sources=[],model_sources=[],kernel_sources=[],
        competition_sources=['arc-prize-2026-arc-agi-2'])
    if gpu_probe:
        code='''import json,subprocess
from pathlib import Path
import torch
report=dict(cuda_available=torch.cuda.is_available(),torch_version=torch.__version__,devices=[])
if report['cuda_available']:
    for i in range(torch.cuda.device_count()):
        p=torch.cuda.get_device_properties(i)
        report['devices'].append(dict(index=i,name=p.name,memory_bytes=p.total_memory,capability=list(torch.cuda.get_device_capability(i))))
Path('/kaggle/working/gpu-probe.json').write_text(json.dumps(report,indent=2))
print('GPU_CAPABILITY_PROBE',json.dumps(report),flush=True)
'''
        metadata.update(id='kunalsinghh03/arc-hydra-gpu-capability-probe',title='ARC Hydra GPU Capability Probe',
            enable_gpu=True,machine_shape='NvidiaTeslaT4',competition_sources=[])
        heading='# GPU capability probe\n\nRead-only device checks before attempting a small ARC-2 neural model. No model/data download, training or competition entry.'
    else:
        files={name:(ROOT/name).read_text(encoding='utf-8') for name in ('arc2/__init__.py','arc2/symbolic.py','arc2/evaluation.py')}
        code='from pathlib import Path\nimport json,time,sys\n'
        code+=f'sources={files!r}\n'
        code+='''for name,source in sources.items():
    path=Path('/kaggle/working')/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source)
sys.path.insert(0,'/kaggle/working')
from arc2.symbolic import solve
from arc2.evaluation import validate_submission
roots=[Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-2'),Path('/kaggle/input/arc-prize-2026-arc-agi-2')]
root=next((p for p in roots if (p/'arc-agi_test_challenges.json').is_file()),None)
if root is None:raise FileNotFoundError('ARC-AGI-2 input challenges not mounted')
challenges=json.loads((root/'arc-agi_test_challenges.json').read_text())
started=time.monotonic();submission={};events=[]
for key,task in sorted(challenges.items()):
    if time.monotonic()-started>3500:
        guesses=[dict(attempt_1=t['input'],attempt_2=t['input']) for t in task['test']]
        info=dict(reason='global budget fallback',consistent_programs=0)
    else:guesses,info=solve(task,seconds=2.0)
    submission[key]=guesses;events.append(dict(task_id=key,**info))
validate_submission(challenges,submission)
Path('/kaggle/working/submission.json').write_text(json.dumps(submission))
report=dict(tasks=len(challenges),test_outputs=sum(len(t['test']) for t in challenges.values()),
    elapsed_seconds=time.monotonic()-started,tasks_with_consistent_program=sum(e.get('consistent_programs',0)>0 for e in events),
    boundary='Symbolic CPU starting baseline; no neural model, weight training or official score established.')
Path('/kaggle/working/arc2-run-metrics.json').write_text(json.dumps(report,indent=2))
Path('/kaggle/working/arc2-search-events.json').write_text(json.dumps(events))
print('ARC2_CPU_BASELINE_COMPLETE',json.dumps(report),flush=True)
'''
        heading=('# ARC-Hydra ARC-AGI-2 CPU starting baseline\n\n'
            'A bounded demonstration-consistent symbolic program search, separate from ARC-AGI-3. '
            'Tries geometry, color mappings, foreground/object crops, scaling, tiling and split-grid combinations. '
            'Two distinct predictions where available; explicit input-copy fallback otherwise. '
            'Uses only each task’s supplied demonstrations, never evaluation-answer files or task-ID lookup. '
            'A reproducible floor, not a claim of competitive performance. No GPU or model training. '
            'Local public-training measurement: 49/1076 test outputs (4.55%); not an unseen or official result.')
    compile(code,'arc2-notebook','exec')
    notebook=dict(nbformat=4,nbformat_minor=4,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3')),
        cells=[dict(cell_type='markdown',metadata={},source=heading),dict(cell_type='code',metadata={},source=code,outputs=[],execution_count=None)])
    output=ROOT/'notebooks'/('gpu_capability_probe' if gpu_probe else 'arc2_symbolic_baseline');output.mkdir(exist_ok=True)
    for name,value in [('submission.ipynb',notebook),('kernel-metadata.json',metadata)]:
        (output/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(kernel=metadata['id'],gpu=gpu_probe,source_sha256=sha256(code.encode()).hexdigest())))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--gpu-probe',action='store_true')
    build(parser.parse_args().gpu_probe)
