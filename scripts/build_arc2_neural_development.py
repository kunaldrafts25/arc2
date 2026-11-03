"""Full-budget public-training-only neural benchmark, separate from submissions."""
from hashlib import sha256
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def build():
    files={name:(ROOT/name).read_text(encoding='utf-8') for name in
        ('arc2/__init__.py','arc2/symbolic.py','arc2/evaluation.py','arc2/neural.py')}
    code='from pathlib import Path\nimport json,time,sys,os\nSTART=time.monotonic()\nEND=START+43200-600\n'
    code+=f'sources={files!r}\n'
    code+='''if os.environ.get('KAGGLE_IS_COMPETITION_RERUN','').strip().lower() in ('1','true'):
    raise RuntimeError('Training-only development notebook; not a competition submission')
for name,source in sources.items():
    path=Path('/kaggle/working')/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source)
sys.path.insert(0,'/kaggle/working')
from arc2.symbolic import solve
from arc2.neural import GridModel
from arc2.evaluation import validate_submission,score_submission
import torch,transformers
print('NEURAL_RUNTIME',dict(torch=torch.__version__,transformers=transformers.__version__,cuda=torch.cuda.is_available()),flush=True)
assert torch.cuda.is_available(), 'Actual CUDA unavailable; stop before loading weights'
print('GPU',torch.cuda.get_device_name(0),torch.cuda.get_device_properties(0).total_memory,flush=True)
roots=[Path('/kaggle/input/datasets/kunalsinghh03/arc2-public-training-development'),Path('/kaggle/input/arc2-public-training-development')]
root=next((p for p in roots if p.is_dir()),None)
if root is None:raise FileNotFoundError('Public training-only development inputs missing')
models=[Path('/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1'),Path('/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1')]
model_path=next((p for p in models if (p/'config.json').is_file()),None)
if model_path is None:raise FileNotFoundError('Pinned public 4B grid model missing')
challenges=json.loads((root/'arc-agi_training_challenges.json').read_text())
model=GridModel(model_path)
print('MODEL_READY',dict(vocab=model.model.config.vocab_size,tokenizer=len(model.tokenizer)),flush=True)
neural={};symbolic={};combined={};events=[];generated=0
def freeze():
    for name,data in [('neural-predictions.json',neural),('symbolic-predictions.json',symbolic),('hybrid-predictions.json',combined)]:
        temporary=Path('/kaggle/working')/(name+'.tmp');temporary.write_text(json.dumps(data));temporary.replace(temporary.with_suffix(''))
    Path('/kaggle/working/neural-events.json').write_text(json.dumps(events))
for number,(key,task) in enumerate(sorted(challenges.items())):
    floor,search=solve(task,seconds=0.25);symbolic[key]=floor
    attempts=[];hybrid=[]
    for index,test in enumerate(task['test']):
        choices=[]
        for orientation in (0,4):
            prediction,info=model.generate(task,index,orientation,deadline=min(END,time.monotonic()+120))
            events.append(dict(task_id=key,test_index=index,**info))
            generated+=info.get('generated_tokens',0)
            if prediction is not None and prediction not in choices:choices.append(prediction)
        if not choices:choices=[floor[index]['attempt_1']]
        if len(choices)==1:choices.append(floor[index]['attempt_2'])
        attempts.append(dict(attempt_1=choices[0],attempt_2=choices[1]))
        first=floor[index]['attempt_1'] if search.get('consistent_programs',0)>0 else choices[0]
        second=next((c for c in choices+[floor[index]['attempt_2']] if c!=first),first)
        hybrid.append(dict(attempt_1=first,attempt_2=second))
    neural[key]=attempts;combined[key]=hybrid
    if (number+1)%5==0:
        freeze();print('NEURAL_PROGRESS',json.dumps(dict(tasks=number+1,total=len(challenges),generated_tokens=generated,elapsed_seconds=time.monotonic()-START)),flush=True)
freeze()
# Targets are opened only after every prediction is frozen. This is public
# TRAINING data, not an unseen evaluation set or official competition score.
solutions=json.loads((root/'arc-agi_training_solutions.json').read_text())
scores={}
for name,data in [('neural',neural),('symbolic',symbolic),('hybrid',combined)]:
    result=score_submission(challenges,solutions,data);result.pop('per_output');scores[name]=result
report=dict(tasks=len(challenges),scores=scores,generated_tokens=generated,elapsed_seconds=time.monotonic()-START,
    boundary='Public training development only; checkpoint may have trained on these tasks. Inference-only baseline, not NVARC TTT reproduction, held-out generalization or official score.')
Path('/kaggle/working/neural-development-metrics.json').write_text(json.dumps(report,indent=2))
print('ARC2_NEURAL_DEVELOPMENT_COMPLETE',json.dumps(report),flush=True)
'''
    compile(code,'arc2-neural-development','exec')
    notebook=dict(nbformat=4,nbformat_minor=4,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3')),
        cells=[dict(cell_type='markdown',metadata={},source='# ARC-Hydra ARC2 pretrained neural development\n\nPublic Apache-2.0 sorokin Qwen3 4B grid checkpoint; formatting informed by NVARC and Jingshui Fu’s Apache-2.0 public notebook. Standard Transformers FP16 inference, optional exact symbolic first attempt, identity/transpose predictions and a digit-grid generation grammar. No weight training, Unsloth or test-time adaptation. Measures public TRAINING tasks only, whose labels the checkpoint may already have seen. Not a competition notebook and no evaluation answers are read. Maximum 12 hours including startup and a ten-minute finalization reserve.'),dict(cell_type='code',metadata={},source=code,outputs=[],execution_count=None)])
    metadata=dict(id='kunalsinghh03/arc-hydra-arc2-neural-development',title='ARC Hydra ARC2 Neural Development',
        code_file='submission.ipynb',language='python',kernel_type='notebook',is_private=True,
        enable_gpu=True,enable_tpu=False,enable_internet=False,machine_shape='NvidiaTeslaT4',
        dataset_sources=['kunalsinghh03/arc2-public-training-development'],competition_sources=[],kernel_sources=[],
        model_sources=['sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1'])
    output=ROOT/'notebooks/arc2_neural_development';output.mkdir(exist_ok=True)
    for name,value in [('submission.ipynb',notebook),('kernel-metadata.json',metadata)]:
        (output/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(kernel=metadata['id'],source_sha256=sha256(code.encode()).hexdigest(),wall_seconds=43200)))


if __name__=='__main__':build()
