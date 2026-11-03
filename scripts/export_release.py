"""Export audited tracked/nonignored source bytes without Git history or private caches."""
import argparse
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED

from scripts.audit_release import ROOT, audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'dist/arc-hydra-source.zip')
    args=parser.parse_args()
    report=audit()
    if not report['passed']:
        print(json.dumps(report,indent=2));raise SystemExit('Release audit failed; nothing exported')
    output=args.output.resolve()
    if output.exists():raise FileExistsError('Refusing to overwrite an existing source export')
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT)
    files=sorted(set(x.decode('utf-8') for x in names.split(b'\0') if x))
    output.parent.mkdir(parents=True,exist_ok=True)
    count=0
    with ZipFile(output,'x',compression=ZIP_DEFLATED,compresslevel=6) as archive:
        for name in files:
            path=ROOT/name
            if path.is_file():archive.write(path,arcname=name);count+=1
    print(json.dumps(dict(output=str(output),files=count,bytes=output.stat().st_size,audit_passed=True),indent=2))


if __name__=='__main__':main()
