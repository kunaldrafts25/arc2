"""Inference-only adapter for the public NVARC-derived 4B grid checkpoint.

Uses the public checkpoint's digit/chat serialization. This does not reproduce
NVARC's LoRA test-time training, DFS decoding or augmentation rescoring.
"""
import time

from .symbolic import grid,orient


def grid_text(value):return '\n'.join(''.join(map(str,row)) for row in grid(value))


def prompt(task,index,orientation=0,skip=0):
    parts=[]
    for pair in task['train'][skip:]:
        parts.extend(('<|im_start|>user\n',grid_text(orient(grid(pair['input']),orientation)),
            '<|im_end|><|im_start|>assistant\n',grid_text(orient(grid(pair['output']),orientation)),'<|im_end|>'))
    parts.extend(('<|im_start|>user\n',grid_text(orient(grid(task['test'][index]['input']),orientation)),
        '<|im_end|><|im_start|>assistant\n'))
    return ''.join(parts)


def parse_output(text):
    text=text.split('<|im_end|>',1)[0].strip('\n')
    if not text or any(c not in '0123456789\n' for c in text):raise ValueError('Expected digit grid only')
    return grid([[int(c) for c in row] for row in text.split('\n')])


def allowed_tokens(suffix,digits,newline,eos):
    if eos in suffix:return [eos]
    rows=[];current=[]
    for token in suffix:
        if token==newline:rows.append(current);current=[]
        elif token in digits:current.append(token)
        else:raise ValueError('Invalid generated grid token')
    if not rows:
        result=list(digits) if len(current)<30 else []
        if current:result.extend((newline,eos))
        return result
    width=len(rows[0])
    if not 1<=width<=30 or any(len(row)!=width for row in rows):raise ValueError('Nonrectangular generated prefix')
    if len(current)<width:return list(digits)
    if len(current)>width:raise ValueError('Generated row too wide')
    return [eos]+([newline] if len(rows)<29 else [])


class GridModel:
    def __init__(self,path):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        if not torch.cuda.is_available():raise RuntimeError('Actual CUDA GPU required before loading neural model')
        self.torch=torch
        self.tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForCausalLM.from_pretrained(str(path),local_files_only=True,trust_remote_code=False,
            use_safetensors=True,torch_dtype=torch.float16,device_map={'':0},low_cpu_mem_usage=True,
            attn_implementation='sdpa')
        self.model.eval()
        self.digits=[self.one_token(str(c)) for c in range(10)]
        self.newline=self.one_token('\n');self.eos=self.tokenizer.eos_token_id
        if self.eos is None:raise ValueError('Tokenizer has no EOS')
        ids=self.digits+[self.newline,self.eos]
        if len(set(ids))!=12 or max(ids)>=self.model.config.vocab_size:
            raise ValueError('Checkpoint grid-token vocabulary mismatch')

    def one_token(self,text):
        ids=self.tokenizer.encode(text,add_special_tokens=False)
        if len(ids)!=1:raise ValueError('Checkpoint requires atomic grid tokens')
        return ids[0]

    def generate(self,task,index,orientation=0,deadline=None):
        if orientation not in (0,4):raise ValueError('Only identity/transpose inference supported')
        if deadline is not None and time.monotonic()>=deadline-5:return None,dict(reason='budget')
        started=time.monotonic();skip=0
        while True:
            ids=self.tokenizer.encode(prompt(task,index,orientation,skip),add_special_tokens=False)
            if len(ids)+960<=8192:break
            if skip>=len(task['train']):return None,dict(reason='context budget')
            skip+=1
        if max(ids)>=self.model.config.vocab_size:raise ValueError('Prompt token exceeds checkpoint vocabulary')
        tokens=self.torch.tensor([ids],device=self.model.device,dtype=self.torch.long)
        options=dict(max_new_tokens=960,do_sample=False,use_cache=True,eos_token_id=self.eos,pad_token_id=self.eos,
            prefix_allowed_tokens_fn=lambda batch,seq:allowed_tokens(seq[len(ids):].tolist(),self.digits,self.newline,self.eos))
        if deadline is not None:options['max_time']=max(1.0,deadline-time.monotonic())
        try:
            with self.torch.inference_mode():output=self.model.generate(input_ids=tokens,**options)
        except self.torch.OutOfMemoryError:
            self.torch.cuda.empty_cache()
            return None,dict(reason='GPU memory exhausted',prompt_tokens=len(ids),orientation=orientation)
        suffix=output[0,len(ids):].tolist()
        info=dict(prompt_tokens=len(ids),generated_tokens=len(suffix),orientation=orientation,
            demonstrations_omitted=skip,elapsed_seconds=time.monotonic()-started,terminated_with_eos=bool(suffix and suffix[-1]==self.eos))
        if not info['terminated_with_eos']:return None,dict(**info,reason='unfinished output')
        try:result=parse_output(self.tokenizer.decode(suffix,skip_special_tokens=False))
        except ValueError:return None,dict(**info,reason='invalid grid')
        if orientation==4:result=orient(result,4)
        return [list(row) for row in result],info
