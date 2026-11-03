"""Private compute mirror of permitted public ARC-2 TRAINING data only."""
from hashlib import sha256
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
output=ROOT/'reference/arc2/training-development';output.mkdir(exist_ok=True)
files=[]
for name in ('arc-agi_training_challenges.json','arc-agi_training_solutions.json'):
    source=ROOT/'reference/arc2/data'/name;target=output/name
    shutil.copyfile(source,target);files.append(dict(name=name,sha256=sha256(target.read_bytes()).hexdigest()))
(output/'manifest.json').write_text(json.dumps(dict(scope='Public training data only; no evaluation answers or test data',files=files),indent=2)+'\n',encoding='utf-8')
metadata=dict(id='kunalsinghh03/arc2-public-training-development',title='ARC2 Public Training Development',
    subtitle='Training-only development inputs for a separate ARC2 baseline',
    description='Only public training challenges and their provided training solution labels. No evaluation-answer files, hidden tasks, credentials or pretrained weights.',licenses=[dict(name='Apache 2.0')])
(output/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
print('Prepared public training-only development inputs')
