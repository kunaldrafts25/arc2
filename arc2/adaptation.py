"""Small, resettable LoRA test-time adaptation on supplied demonstrations only."""
import random
import time

from .neural import grid_text
from .symbolic import grid,orient


def augmented_pair(pair,rng):
    orientation=rng.randrange(8);permutation=list(range(10));rng.shuffle(permutation)
    def transform(value):
        return [[permutation[c] for c in row] for row in orient(grid(value),orientation)]
    return dict(input=transform(pair['input']),output=transform(pair['output']))


def training_tokens(pair,tokenizer):
    prefix='<|im_start|>user\n'+grid_text(pair['input'])+'<|im_end|><|im_start|>assistant\n'
    suffix=grid_text(pair['output'])+'<|im_end|>'
    start=tokenizer.encode(prefix,add_special_tokens=False)
    full=tokenizer.encode(prefix+suffix,add_special_tokens=False)
    if full[:len(start)]!=start:raise ValueError('Tokenizer prefix boundary changed')
    if len(full)>2048:raise ValueError('Training pair exceeds adaptation context')
    return full,[-100]*len(start)+full[len(start):]


class TaskAdapter:
    def __init__(self,backend,rank=16):
        from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
        self.backend=backend;self.torch=backend.torch
        self.get_state=get_peft_model_state_dict;self.set_state=set_peft_model_state_dict
        config=LoraConfig(r=rank,lora_alpha=32,lora_dropout=0,bias='none',task_type='CAUSAL_LM',
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
        backend.model=get_peft_model(backend.model,config)
        trainable=[(n,p) for n,p in backend.model.named_parameters() if p.requires_grad]
        if not trainable or any('lora_' not in n for n,p in trainable):raise ValueError('Only LoRA parameters may train')
        self.parameters=[p for n,p in trainable]
        self.trainable_parameters=sum(p.numel() for p in self.parameters)
        if self.trainable_parameters>50000000:raise ValueError('Adaptation parameter budget exceeded')
        self.initial={k:v.detach().cpu().clone() for k,v in self.get_state(backend.model).items()}
        backend.model.enable_input_require_grads()
        self.reset()

    def reset(self):
        self.set_state(self.backend.model,self.initial)
        self.backend.model.eval()

    def adapt(self,task,steps=8,deadline=None):
        self.reset();model=self.backend.model;torch=self.torch
        started=time.monotonic();rng=random.Random(42);losses=[];completed=0
        optimizer=torch.optim.AdamW(self.parameters,lr=5e-5,weight_decay=0,foreach=False)
        scaler=torch.amp.GradScaler('cuda',init_scale=1024)
        model.config.use_cache=False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.train()
        reason='complete'
        try:
            for step in range(steps):
                if deadline is not None and time.monotonic()>=deadline-3:reason='budget';break
                pair=augmented_pair(task['train'][step%len(task['train'])],rng)
                ids,labels=training_tokens(pair,self.backend.tokenizer)
                x=torch.tensor([ids],device=model.device,dtype=torch.long)
                y=torch.tensor([labels],device=model.device,dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):loss=model(input_ids=x,labels=y).loss
                if not bool(torch.isfinite(loss)):reason='nonfinite loss';break
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters,1.0)
                old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                if scaler.get_scale()>=old_scale:completed+=1
                losses.append(float(loss.detach().cpu()))
        except torch.OutOfMemoryError:
            reason='GPU memory exhausted'
        finally:
            optimizer.zero_grad(set_to_none=True);del optimizer
            model.gradient_checkpointing_disable();model.config.use_cache=True;model.eval()
            torch.cuda.empty_cache()
        return dict(steps_requested=steps,steps_completed=completed,losses=losses,stop_reason=reason,
            elapsed_seconds=time.monotonic()-started,trainable_parameters=self.trainable_parameters,
            source='Supplied demonstration input/output pairs only; no test targets')

    def change_from_initial(self):
        current=self.get_state(self.backend.model)
        return max(float((v.detach().cpu()-self.initial[k]).abs().max()) for k,v in current.items())
