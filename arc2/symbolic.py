"""Bounded task-local program search; uses only the task's demonstration pairs."""
from collections import Counter,deque
import time


def grid(value):
    if not isinstance(value,(list,tuple)) or not 1<=len(value)<=30:raise ValueError('Invalid grid height')
    rows=tuple(tuple(row) for row in value)
    if not rows[0] or len(rows[0])>30 or any(len(row)!=len(rows[0]) for row in rows):raise ValueError('Invalid grid width')
    if any(type(c) is not int or not 0<=c<=9 for row in rows for c in row):raise ValueError('Invalid grid colors')
    return rows


def modal(g):return Counter(c for row in g for c in row).most_common(1)[0][0]


def rotate(g):return tuple(tuple(row) for row in zip(*g[::-1]))


def orient(g,index):
    if index>=4:g=tuple(tuple(row) for row in zip(*g));index-=4
    for _ in range(index):g=rotate(g)
    return g


def crop(g,color=None,background=None):
    bg=modal(g) if background is None else background
    points=[(r,c) for r,row in enumerate(g) for c,x in enumerate(row) if (x==color if color is not None else x!=bg)]
    if not points:return None
    r0=min(p[0] for p in points);r1=max(p[0] for p in points)+1
    c0=min(p[1] for p in points);c1=max(p[1] for p in points)+1
    return tuple(row[c0:c1] for row in g[r0:r1])


def object_crop(g,selection,monochrome,background):
    bg=modal(g) if background is None else background
    seen=set();objects=[];h=len(g);w=len(g[0])
    for r in range(h):
        for c in range(w):
            if (r,c) in seen or g[r][c]==bg:continue
            pending=deque([(r,c)]);seen.add((r,c));points=[]
            while pending:
                y,x=pending.popleft();points.append((y,x))
                for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                    if 0<=yy<h and 0<=xx<w and (yy,xx) not in seen and g[yy][xx]!=bg and (not monochrome or g[yy][xx]==g[r][c]):
                        seen.add((yy,xx));pending.append((yy,xx))
            objects.append(points)
    if not objects:return None
    objects.sort(key=lambda p:(len(p),min(p)))
    points=objects[-1] if selection=='largest' else objects[0]
    r0=min(p[0] for p in points);r1=max(p[0] for p in points)+1
    c0=min(p[1] for p in points);c1=max(p[1] for p in points)+1
    return tuple(row[c0:c1] for row in g[r0:r1])


def resize(g,scale,repeat=False):
    if len(g)*scale>30 or len(g[0])*scale>30:return None
    if repeat:return tuple(row*scale for _ in range(scale) for row in g)
    return tuple(tuple(c for x in row for c in [x]*scale) for row in g for _ in range(scale))


def reduce_grid(g,scale):
    if len(g)%scale or len(g[0])%scale:return None
    result=[]
    for r in range(0,len(g),scale):
        row=[]
        for c in range(0,len(g[0]),scale):
            values={g[y][x] for y in range(r,r+scale) for x in range(c,c+scale)}
            if len(values)!=1:return None
            row.append(next(iter(values)))
        result.append(tuple(row))
    return tuple(result)


def compress(g):
    rows=[]
    for row in g:
        if not rows or row!=rows[-1]:rows.append(row)
    columns=[]
    for col in zip(*rows):
        if not columns or col!=columns[-1]:columns.append(col)
    return tuple(tuple(row) for row in zip(*columns))


def halves(g,axis,mode,background):
    if axis:g=orient(g,4)
    n=len(g);middle=n//2
    if n<2:return None
    first=g[:middle];second=g[middle+n%2:]
    bg=modal(g) if background is None else background
    if mode=='first':out=first
    elif mode=='second':out=second
    else:
        operations={'or':lambda a,b:a or b,'and':lambda a,b:a and b,'xor':lambda a,b:a!=b}
        op=operations[mode]
        out=tuple(tuple(int(op(a!=bg,b!=bg)) for a,b in zip(ra,rb)) for ra,rb in zip(first,second))
    return orient(out,4) if axis else out


def operations(colors):
    yield 'identity',0,lambda g:g
    yield 'compress_repeated_rows_columns',1,compress
    for bg in (None,0):
        yield f'foreground_bbox_{bg}',1,lambda g,bg=bg:crop(g,background=bg)
        for selection in ('largest','smallest'):
            for mono in (False,True):
                yield f'{selection}_object_{mono}_{bg}',2,lambda g,s=selection,m=mono,b=bg:object_crop(g,s,m,b)
        for axis in (0,1):
            for mode in ('first','second','or','and','xor'):
                yield f'halves_{axis}_{mode}_{bg}',2,lambda g,a=axis,m=mode,b=bg:halves(g,a,m,b)
    for color in colors:yield f'color_bbox_{color}',2,lambda g,c=color:crop(g,color=c)
    for scale in (2,3,4):
        yield f'upscale_{scale}',1,lambda g,s=scale:resize(g,s)
        yield f'tile_{scale}',2,lambda g,s=scale:resize(g,s,True)
        yield f'uniform_downscale_{scale}',1,lambda g,s=scale:reduce_grid(g,s)


def fit_mapping(inputs,outputs):
    mapping={}
    for before,after in zip(inputs,outputs):
        if before is None or len(before)!=len(after) or len(before[0])!=len(after[0]):return None
        for ra,rb in zip(before,after):
            for a,b in zip(ra,rb):
                if a in mapping and mapping[a]!=b:return None
                mapping[a]=b
    return mapping


def solve(task,seconds=1.0):
    started=time.monotonic();examples=task['train'];tests=[grid(t['input']) for t in task['test']]
    if not examples:return [dict(attempt_1=[list(row) for row in g],attempt_2=[list(row) for row in g]) for g in tests],dict(programs=0,reason='no demonstrations')
    inputs=[grid(t['input']) for t in examples];outputs=[grid(t['output']) for t in examples]
    colors=sorted({c for g in inputs for row in g for c in row})
    mappings={};prediction_sets=set();candidates=[];tried=0;timed_out=False
    for name,cost,operation in operations(colors):
        if time.monotonic()-started>=seconds:timed_out=True;break
        transformed=[operation(g) for g in inputs]
        if any(g is None for g in transformed):continue
        for orientation in range(8):
            views=tuple(orient(g,orientation) for g in transformed)
            tried+=1
            if views not in mappings:mappings[views]=fit_mapping(views,outputs)
            mapping=mappings[views]
            if mapping is None:continue
            predictions=[]
            for g in tests:
                result=operation(g)
                if result is None:break
                result=orient(result,orientation)
                result=tuple(tuple(mapping.get(c,c) for c in row) for row in result)
                predictions.append(result)
            if len(predictions)!=len(tests):continue
            signature=tuple(predictions)
            if signature in prediction_sets:continue
            prediction_sets.add(signature)
            recolors=sum(a!=b for a,b in mapping.items())
            score=cost+0.1*(orientation!=0)+0.05*recolors
            candidates.append((score,name,orientation,tuple(predictions)))
    candidates.sort(key=lambda c:(c[0],c[1],c[2]))
    attempts=[]
    for i,g in enumerate(tests):
        unique=[]
        for candidate in candidates:
            prediction=candidate[3][i]
            if prediction not in unique:unique.append(prediction)
            if len(unique)==2:break
        if not unique:unique=[g]
        if len(unique)==1:unique.append(g if g!=unique[0] else tuple(tuple(0 for _ in row) for row in g))
        attempts.append(dict(attempt_1=[list(row) for row in unique[0]],attempt_2=[list(row) for row in unique[1]]))
    return attempts,dict(programs=tried,consistent_programs=len(candidates),timed_out=timed_out,
        elapsed_seconds=time.monotonic()-started,selected=[dict(name=c[1],orientation=c[2],complexity=c[0]) for c in candidates[:2]])
