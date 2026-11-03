"""Build an original-account ARC2 candidate with a bounded visible-input preview."""
from hashlib import sha256
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(version=1):
    files = {name: (ROOT/name).read_text(encoding='utf-8') for name in (
        'arc2/__init__.py', 'arc2/symbolic.py', 'arc2/evaluation.py', 'arc2/neural.py',
        'arc2/bounded_inference.py')}
    if version == 1:
        raise ValueError('Version 1 is frozen. Use --version 2 for the targeted validation.')
    code = 'from pathlib import Path\nimport json,time,sys,os\nSTART=time.monotonic()\n'
    code += f'sources={files!r}\n'
    code += '''
RERUN=os.environ.get('KAGGLE_IS_COMPETITION_RERUN','').strip().lower() in ('1','true')
END=START+(43200-600 if RERUN else 500)
for name,source in sources.items():
    p=Path('/kaggle/working')/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)
sys.path.insert(0,'/kaggle/working')
from arc2.symbolic import solve
from arc2.evaluation import validate_submission
from arc2.neural import GridModel
from arc2.bounded_inference import generate_chunked,run
import torch,transformers
roots=[Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-2'),Path('/kaggle/input/arc-prize-2026-arc-agi-2')]
root=next((p for p in roots if (p/'arc-agi_test_challenges.json').is_file()),None)
if root is None:raise FileNotFoundError('Actual competition test inputs not mounted')
challenges=json.loads((root/'arc-agi_test_challenges.json').read_text())
submission={key:solve(task,seconds=0.25)[0] for key,task in sorted(challenges.items())}
events=[]
def save(predictions,records):
    submission.update(predictions)
    validate_submission(challenges,submission)
    p=Path('/kaggle/working/submission.json.tmp');p.write_text(json.dumps(submission));p.replace(p.with_suffix(''))
    Path('/kaggle/working/inference-events.json').write_text(json.dumps(records))
save({},events)
models=[Path('/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1'),Path('/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1')]
model_path=next((p for p in models if (p/'config.json').is_file()),None)
if model_path is None:raise FileNotFoundError('Pinned public checkpoint missing')
model=GridModel(model_path)
print('ARC2_MODEL_READY',dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,transformers=transformers.__version__,rerun=RERUN),flush=True)
# Fixed public TRAINING inputs chosen by event failure/length, never by labels.
# Small-prompt numerical parity must hold before using the changed decoder.
training=json.loads((root/'arc-agi_training_challenges.json').read_text())
checks=[]
for key,view in [('00576224',0),('007bbfb7',4)]:
    task=training[key]
    before,binfo=model.generate(task,0,view,deadline=min(END,time.monotonic()+25))
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    after,ainfo=generate_chunked(model,task,0,view,deadline=min(END,time.monotonic()+25))
    checks.append(dict(task_id=key,orientation=view,equal=before is not None and before==after,
                       baseline=binfo,chunked=ainfo,peak_bytes=torch.cuda.max_memory_allocated()))
    print('PARITY',json.dumps(checks[-1]),flush=True)
use_chunked=all(c['equal'] for c in checks)
memory_check=None
if not RERUN and use_chunked and time.monotonic()<END-260:
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    prediction,info=generate_chunked(model,training['05a7bcf2'],0,deadline=min(END-40,time.monotonic()+240))
    memory_check=dict(task_id='05a7bcf2',valid=prediction is not None,info=info,
                      peak_bytes=torch.cuda.max_memory_allocated())
    print('LONG_PROMPT_CHECK',json.dumps(memory_check),flush=True)
def baseline_generator(m,task,index,orientation,deadline):
    return m.generate(task,index,orientation,deadline)
# Visible preview uses a small actual-test subset to fit the remaining free quota.
# Official rerun processes ALL supplied test inputs; fallback was written first.
active=challenges if RERUN else {k:challenges[k] for k in sorted(challenges)[:4]}
_,events=run(active,model,END,save,generator=generate_chunked if use_chunked else baseline_generator)
save({},events)
report=dict(rerun=RERUN,elapsed_seconds=time.monotonic()-START,tasks=len(challenges),
    test_outputs=sum(len(t['test']) for t in challenges.values()),active_neural_tasks=len(active),
    valid_generations=sum(not e.get('reason') for e in events),events=len(events),
    decoder='chunked' if use_chunked else 'baseline',parity_checks=checks,memory_check=memory_check,
    boundary='Actual competition-input inference. Visible preview capped below ten minutes; official rerun capped below twelve hours. No solution files read. Functional and memory diagnostics do not establish accuracy or an official score.')
Path('/kaggle/working/competition-run.json').write_text(json.dumps(report,indent=2))
print('ARC2_COMPETITION_COMPLETE',json.dumps(report),flush=True)
'''
    compile(code, 'arc2-competition', 'exec')
    notebook = dict(nbformat=4, nbformat_minor=4,
        metadata=dict(kernelspec=dict(name='python3', display_name='Python 3')),
        cells=[dict(cell_type='markdown', metadata={}, source='# ARC2 inference-only hybrid candidate\n\nPublic Apache-2.0 Sorokin grid model. Demonstration-consistent symbolic candidates retain first-attempt priority. Identity and transpose decoding, rectangular grammar, coverage-first scheduling and chunked KV-cached prefill with a baseline parity gate. No adaptation or answer-file access. Bounded visible-input preview; full-budget official rerun. Training-development hybrid: 281/1076, potentially contaminated by checkpoint training; no held-out improvement claimed.'),
               dict(cell_type='code', metadata={}, source=code, outputs=[], execution_count=None)])
    metadata = dict(id='kunalsinghh25/arc-hydra-arc2-inference-hybrid', title='ARC Hydra ARC2 Inference Hybrid',
        code_file='submission.ipynb', language='python', kernel_type='notebook', is_private=True,
        enable_gpu=True, enable_tpu=False, enable_internet=False, machine_shape='NvidiaTeslaT4',
        dataset_sources=[], competition_sources=['arc-prize-2026-arc-agi-2'], kernel_sources=[],
        model_sources=['sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1'])
    output = ROOT/f'notebooks/arc2_inference_hybrid_v{version}'
    output.mkdir(exist_ok=True)
    for name, value in [('submission.ipynb', notebook), ('kernel-metadata.json', metadata)]:
        (output/name).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    manifest = dict(kernel=metadata['id'], source_sha256=sha256(code.encode()).hexdigest(),
                    files={name:sha256((output/name).read_bytes()).hexdigest() for name in ('kernel-metadata.json','submission.ipynb')},
                    preview_seconds=500, official_seconds=42600)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',type=int,choices=[2],default=2)
    build(parser.parse_args().version)
