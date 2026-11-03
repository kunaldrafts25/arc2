"""Short actual-GPU probe of resettable task-local LoRA adaptation."""
from hashlib import sha256
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def build():
    files={name:(ROOT/name).read_text(encoding='utf-8') for name in ('arc2/__init__.py','arc2/symbolic.py','arc2/neural.py','arc2/adaptation.py')}
    code='from pathlib import Path\nimport json,time,sys\nSTART=time.monotonic()\nEND=START+600-60\n'
    code+=f'sources={files!r}\n'
    code+='''for name,source in sources.items():
    path=Path('/kaggle/working')/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source)
sys.path.insert(0,'/kaggle/working')
# Plain FP16 LoRA does not use the optional TorchAO quantization backend.
# The stock image has TorchAO 0.10, which current PEFT rejects during import.
import subprocess
subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=True,stdout=subprocess.DEVNULL)
import torch,transformers,peft
print('ADAPTATION_RUNTIME',dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,cuda=torch.cuda.is_available()),flush=True)
assert torch.cuda.is_available(), 'No real CUDA GPU'
from arc2.neural import GridModel
from arc2.adaptation import TaskAdapter
roots=[Path('/kaggle/input/datasets/kunalsinghh03/arc2-public-training-development'),Path('/kaggle/input/arc2-public-training-development')]
root=next(p for p in roots if p.is_dir())
models=[Path('/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1'),Path('/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1')]
model_path=next(p for p in models if (p/'config.json').is_file())
challenges=json.loads((root/'arc-agi_training_challenges.json').read_text())
backend=GridModel(model_path);adapter=TaskAdapter(backend,rank=16)
print('ADAPTER_READY',adapter.trainable_parameters,flush=True)
cases=[]
for key in sorted(challenges)[:2]:
    task=challenges[key]
    info=adapter.adapt(task,steps=8,deadline=min(END,time.monotonic()+90))
    delta=adapter.change_from_initial()
    prediction,generation=backend.generate(task,0,deadline=min(END,time.monotonic()+90))
    adapter.reset();reset_delta=adapter.change_from_initial()
    case=dict(task_id=key,adaptation=info,adapter_delta=delta,reset_delta=reset_delta,
        valid_prediction=prediction is not None,generation=generation,prediction=prediction)
    cases.append(case);print('ADAPTATION_CASE',json.dumps(case),flush=True)
report=dict(passed=all(c['adaptation']['steps_completed']>0 and c['adapter_delta']>0 and c['reset_delta']==0 and c['valid_prediction'] for c in cases),
    cases=cases,max_gpu_memory_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-START,
    boundary='Actual GPU functional probe on two public training tasks. Only demonstration pairs used for adaptation; no test targets read. Not an accuracy comparison or official result.')
Path('/kaggle/working/adaptation-probe.json').write_text(json.dumps(report,indent=2))
print('ARC2_ADAPTATION_PROBE_COMPLETE',json.dumps(report),flush=True)
'''
    compile(code,'arc2-adaptation-probe','exec')
    notebook=dict(nbformat=4,nbformat_minor=4,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3')),
        cells=[dict(cell_type='markdown',metadata={},source='# ARC2 task-local LoRA GPU probe\n\nTests finite demonstration-only learning, actual adapter changes, exact per-task reset, bounded GPU memory and structured generation on two public training tasks. Rank 16, up to eight steps per task, maximum ten minutes. No base-model training, no test target labels, no evaluation-answer files or competition submission.'),dict(cell_type='code',metadata={},source=code,outputs=[],execution_count=None)])
    metadata=json.loads((ROOT/'notebooks/arc2_neural_development/kernel-metadata.json').read_text(encoding='utf-8'))
    metadata.update(id='kunalsinghh03/arc-hydra-arc2-adaptation-probe',title='ARC Hydra ARC2 Adaptation Probe')
    output=ROOT/'notebooks/arc2_adaptation_probe';output.mkdir(exist_ok=True)
    for name,value in [('submission.ipynb',notebook),('kernel-metadata.json',metadata)]:
        (output/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(kernel=metadata['id'],source_sha256=sha256(code.encode()).hexdigest(),wall_seconds=600)))


if __name__=='__main__':build()
