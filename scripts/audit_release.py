"""Audit tracked and nonignored files for a source-only GitHub release.

Prints paths and finding types, never matching secret values. Does not stage,
delete, upload or rewrite any file. Python standard library and git only.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
WEIGHT_SUFFIXES={'.safetensors','.gguf','.pt','.pth','.ckpt'}
PATTERNS={
    'Kaggle access token':re.compile(rb'KGAT_([A-Za-z0-9_-]{32,})'),
    'GitHub token':re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})'),
    'AWS access key':re.compile(rb'AKIA[0-9A-Z]{16}'),
    'private key':re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'signed download URL':re.compile(rb'(?:X-(?:Goog|Amz)-Signature|GoogleAccessId|X-Goog-Credential)='),
}


def findings(path, data, max_bytes, known_secret=b''):
    result=[]
    parts=Path(path).parts
    if '.kaggle' in parts or Path(path).name in {'.env','access_token','kaggle.json'}:
        result.append('credential file')
    if Path(path).suffix.lower() in WEIGHT_SUFFIXES:result.append('model weights are not part of the code release')
    if len(data)>max_bytes:result.append('file exceeds release size limit')
    if known_secret and known_secret in data:result.append('matches project-local credential')
    for label,pattern in PATTERNS.items():
        for match in pattern.finditer(data):
            if label=='Kaggle access token' and set(match.group(1).lower())=={ord('x')}:continue
            result.append(label);break
    return sorted(set(result))


def audit(max_file_mib=10):
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT)
    paths=sorted(set(p.decode('utf-8') for p in names.split(b'\0') if p))
    credential=ROOT/'.kaggle/access_token'
    secret=credential.read_text(encoding='utf-8-sig').strip().encode() if credential.exists() else b''
    issues=[];total=count=0
    for name in paths:
        path=ROOT/name
        if not path.is_file():continue
        data=path.read_bytes();total+=len(data);count+=1
        found=findings(name,data,max_file_mib*1024**2,secret)
        if found:issues.append(dict(path=name,findings=found))
    return dict(passed=not issues,files=count,total_bytes=total,total_mib=round(total/1024**2,2),
        max_file_mib=max_file_mib,issues=issues,
        scope='Tracked plus nonignored working-tree files; credentials are never printed. Does not certify dependencies, licenses or old Git object contents.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-file-mib',type=int,default=10)
    args=parser.parse_args()
    if args.max_file_mib<1:parser.error('Size limit must be positive')
    result=audit(args.max_file_mib)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
