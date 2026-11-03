"""Chunked prefill and coverage-first inference, separate from frozen baselines."""
import time

from .neural import allowed_tokens, parse_output, prompt
from .symbolic import grid, orient, solve
from .evaluation import validate_submission


def select_hybrid(floor, consistent, candidates):
    """Retain the measured baseline selection rule and distinct second guess."""
    choices = []
    for candidate in candidates:
        if candidate not in choices:
            choices.append(candidate)
    if not choices:
        choices = [floor['attempt_1']]
    if len(choices) == 1:
        choices.append(floor['attempt_2'])
    first = floor['attempt_1'] if consistent else choices[0]
    second = next((c for c in choices+[floor['attempt_2']] if c != first), first)
    return dict(attempt_1=first, attempt_2=second)


def decode_tokens(model, ids, deadline, chunk_size=256):
    """Causal KV-cached greedy decoding; never allocate a full prompt attention map."""
    torch = model.torch
    cache = None
    suffix = []
    with torch.inference_mode():
        for start in range(0, len(ids), chunk_size):
            if time.monotonic() >= deadline:
                return suffix, 'prefill deadline'
            tokens = torch.tensor([ids[start:start+chunk_size]], device=model.model.device, dtype=torch.long)
            output = model.model(input_ids=tokens, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            logits = output.logits[0, -1].clone()
            del output
        for _ in range(960):
            if time.monotonic() >= deadline:
                return suffix, 'decode deadline'
            allowed = allowed_tokens(suffix, model.digits, model.newline, model.eos)
            # Preserve the raw model ranking among grammar-allowed tokens.
            token = allowed[int(logits[allowed].argmax().item())]
            suffix.append(token)
            if token == model.eos:
                return suffix, 'complete'
            output = model.model(input_ids=torch.tensor([[token]], device=model.model.device, dtype=torch.long),
                                 past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            logits = output.logits[0, -1].clone()
            del output
    return suffix, 'token limit'


def generate_chunked(model, task, index, orientation=0, deadline=None, chunk_size=256):
    if orientation not in (0, 4):
        raise ValueError('Only measured identity/transpose views supported')
    started = time.monotonic()
    deadline = deadline if deadline is not None else started+120
    if started >= deadline-1:
        return None, dict(reason='budget', orientation=orientation)
    skip = 0
    while True:
        ids = model.tokenizer.encode(prompt(task, index, orientation, skip), add_special_tokens=False)
        if len(ids)+960 <= 8192:
            break
        if skip >= len(task['train']):
            return None, dict(reason='context budget', orientation=orientation)
        skip += 1
    if not ids or max(ids) >= model.model.config.vocab_size:
        raise ValueError('Prompt vocabulary mismatch')
    info = dict(prompt_tokens=len(ids), orientation=orientation, demonstrations_omitted=skip,
                chunk_size=chunk_size)
    try:
        suffix, stop = decode_tokens(model, ids, deadline, chunk_size)
    except model.torch.OutOfMemoryError:
        # decode_tokens has unwound, releasing its live tensors and cache first.
        model.torch.cuda.empty_cache()
        return None, dict(**info, reason='GPU memory exhausted', elapsed_seconds=time.monotonic()-started)
    info.update(generated_tokens=len(suffix), stop_reason=stop, elapsed_seconds=time.monotonic()-started,
                terminated_with_eos=bool(suffix and suffix[-1] == model.eos))
    if not info['terminated_with_eos']:
        return None, dict(**info, reason='unfinished output')
    try:
        result = parse_output(model.tokenizer.decode(suffix, skip_special_tokens=False))
    except ValueError:
        return None, dict(**info, reason='invalid grid')
    if orientation == 4:
        result = orient(result, 4)
    return [list(row) for row in result], info


def run(challenges, model, deadline, save, generator=generate_chunked, clock=time.monotonic):
    """Complete fallback first, then one view across all outputs before the next."""
    floors = {}
    consistent = {}
    candidates = {}
    submission = {}
    events = []
    for key, task in sorted(challenges.items()):
        floor, info = solve(task, seconds=0.25 if clock() < deadline else 0)
        floors[key] = floor
        consistent[key] = info.get('consistent_programs', 0) > 0
        candidates[key] = [[] for _ in task['test']]
        submission[key] = [dict(attempt) for attempt in floor]
    validate_submission(challenges, submission)
    save(submission, events)
    work = [(key, i) for key in sorted(challenges) for i in range(len(challenges[key]['test']))]
    def cost(key, index):
        task = challenges[key]
        output_tokens = max((len(p['output'])*(len(p['output'][0])+1) for p in task['train']), default=930)
        context_chars = len(prompt(task, index))
        return max(30, output_tokens)*(1+context_chars/2000)
    costs = [cost(key, index) for key, index in work]
    for view in (0, 4):
        remaining_cost = sum(costs)
        for position, (key, index) in enumerate(work):
            remaining = deadline-clock()
            if remaining <= 2:
                break
            # Long prompts and large demonstration outputs need more decode time.
            # Reserve a weighted share for each remaining output, cap at 300s.
            share = min(300, max(2, remaining*costs[position]/remaining_cost))
            remaining_cost -= costs[position]
            prediction, info = generator(model, challenges[key], index, orientation=view,
                                         deadline=min(deadline, clock()+share))
            events.append(dict(task_id=key, test_index=index, **info))
            if prediction is not None:
                candidates[key][index].append(prediction)
            submission[key][index] = select_hybrid(floors[key][index], consistent[key], candidates[key][index])
            validate_submission(challenges, submission)
            save(submission, events)
    return submission, events
