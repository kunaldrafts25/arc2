"""Restore release artifacts to their recorded paths, verifying every SHA-256."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
from urllib.request import urlopen
from urllib.parse import urlsplit


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(4*1024**2):h.update(block)
    return h.hexdigest()


def target(root,name):
    path=(root/name).resolve()
    if path==root or not path.is_relative_to(root):raise ValueError('Artifact path escapes destination')
    if os.name=='nt' and not str(path).startswith('\\\\?\\'):
        value=str(path)
        path=Path('\\\\?\\UNC\\'+value[2:] if value.startswith('\\\\') else '\\\\?\\'+value)
    return path


def download(asset,cache):
    path=target(cache,asset['name'])
    if path.exists() and path.stat().st_size==asset['bytes'] and sha256(path)==asset['sha256']:return path
    partial=path.with_suffix(path.suffix+'.partial')
    url=urlsplit(asset['url'])
    if url.scheme!='https' or url.hostname not in {'github.com','api.github.com'}:
        raise ValueError('Expected an HTTPS GitHub release asset URL')
    print('Downloading',asset['name'],flush=True)
    with urlopen(asset['url'],timeout=120) as source,partial.open('wb') as dest:
        shutil.copyfileobj(source,dest,length=4*1024**2)
    if partial.stat().st_size!=asset['bytes'] or sha256(partial)!=asset['sha256']:
        raise ValueError('Downloaded artifact failed size/hash check: '+asset['name'])
    partial.replace(path)
    return path


def restore(manifest,root,cache,kind='all',verify_only=False,overwrite=False):
    root=root.resolve();cache=cache.resolve()
    selected=[r for r in manifest['files'] if kind=='all' or (kind=='adapters')==r['kind'].endswith('adapter')]
    pending=[]
    for row in selected:
        path=target(root,row['path'])
        if path.is_file() and path.stat().st_size==row['bytes'] and sha256(path)==row['sha256']:continue
        if verify_only:raise ValueError('Missing or different artifact: '+row['path'])
        if path.exists() and not overwrite:raise FileExistsError('Different local file; use --overwrite explicitly: '+row['path'])
        pending.append(row)
    if verify_only or not pending:return dict(verified=len(selected),restored=0)
    cache.mkdir(parents=True,exist_ok=True)
    byasset={}
    for row in pending:byasset.setdefault(row['asset'],[]).append(row)
    assets={a['name']:a for a in manifest['assets']};restored=0
    for name,rows in byasset.items():
        payload=download(assets[name],cache)
        if assets[name]['kind']=='raw-data-tar-gzip':
            members={row['archive_member']:row for row in rows};found=set()
            with tarfile.open(payload,'r|gz') as archive:
                for member in archive:
                    if member.name not in members:continue
                    if not member.isfile():raise ValueError('Expected a regular archive file')
                    row=members[member.name];path=target(root,row['path']);path.parent.mkdir(parents=True,exist_ok=True)
                    partial=path.with_suffix(path.suffix+'.restoring')
                    with archive.extractfile(member) as src,partial.open('wb') as dst:shutil.copyfileobj(src,dst,length=4*1024**2)
                    if partial.stat().st_size!=row['bytes'] or sha256(partial)!=row['sha256']:raise ValueError('Restored hash differs: '+row['path'])
                    partial.replace(path);found.add(member.name);restored+=1
            if found!=set(members):raise ValueError('Archive did not contain every requested artifact')
        else:
            for row in rows:
                path=target(root,row['path']);path.parent.mkdir(parents=True,exist_ok=True)
                partial=path.with_suffix(path.suffix+'.restoring');shutil.copyfile(payload,partial)
                if sha256(partial)!=row['sha256']:raise ValueError('Adapter hash mismatch')
                partial.replace(path);restored+=1
    return dict(verified=len(selected),restored=restored)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=Path(__file__).resolve().parents[1]/'ARTIFACTS.json')
    parser.add_argument('--dest',type=Path)
    parser.add_argument('--kind',choices=['all','adapters','raw'],default='all')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args();root=(args.dest or args.manifest.resolve().parent).resolve()
    result=restore(json.loads(args.manifest.read_text(encoding='utf-8')),root,root/'.artifact-cache',args.kind,args.verify_only,args.overwrite)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
