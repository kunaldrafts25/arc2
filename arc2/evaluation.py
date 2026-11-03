"""Official two-attempt exact-match metric, averaged over test outputs."""
from .symbolic import grid


def validate_submission(challenges,submission):
    if set(challenges)!=set(submission):raise ValueError('Submission IDs must exactly match challenge IDs')
    for key,task in challenges.items():
        guesses=submission[key]
        if not isinstance(guesses,list) or len(guesses)!=len(task['test']):raise ValueError('Test-output count mismatch: '+key)
        for attempt in guesses:
            if not isinstance(attempt,dict) or set(attempt)!=set(('attempt_1','attempt_2')):
                raise ValueError('Exactly two named attempts required')
            grid(attempt['attempt_1']);grid(attempt['attempt_2'])


def score_submission(challenges,solutions,submission):
    validate_submission(challenges,submission)
    if set(challenges)!=set(solutions):raise ValueError('Solution IDs differ')
    results=[]
    for key,task in challenges.items():
        if len(solutions[key])!=len(task['test']):raise ValueError('Solution-output count mismatch')
        for index,target in enumerate(solutions[key]):
            target=grid(target);attempt=submission[key][index]
            correct=any(grid(attempt[name])==target for name in ('attempt_1','attempt_2'))
            results.append(dict(task_id=key,test_index=index,correct=correct))
    return dict(correct_outputs=sum(r['correct'] for r in results),test_outputs=len(results),
        score=100*sum(r['correct'] for r in results)/len(results) if results else 0,per_output=results)
