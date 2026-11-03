from copy import deepcopy

from arc2.bounded_inference import run, select_hybrid
from arc2.evaluation import validate_submission


def test_hybrid_preserves_baseline_precedence_and_distinct_guesses():
    floor = dict(attempt_1=[[1]], attempt_2=[[2]])
    assert select_hybrid(floor, True, [[[3]], [[4]]]) == dict(attempt_1=[[1]], attempt_2=[[3]])
    assert select_hybrid(floor, False, [[[3]], [[3]]]) == dict(attempt_1=[[3]], attempt_2=[[2]])
    assert select_hybrid(floor, False, []) == floor


def test_every_output_gets_first_view_before_any_second_view_and_fallback_is_durable():
    challenges = {k: dict(train=[], test=[dict(input=[[0]]) for _ in range(n)])
                  for k, n in [('b', 1), ('a', 2)]}
    original = deepcopy(challenges)
    calls = []
    saved = []
    def generator(model, task, index, orientation, deadline):
        calls.append((index, orientation))
        return [[3 if orientation == 0 else 4]], dict(orientation=orientation)
    def save(submission, events):
        validate_submission(challenges, submission)
        saved.append(deepcopy(submission))
    submission, _ = run(challenges, None, 100, save, generator=generator, clock=lambda: 0)
    assert [view for _, view in calls] == [0, 0, 0, 4, 4, 4]
    assert saved[0]['a'][0]['attempt_1'] == [[0]]
    assert submission['a'][0] == dict(attempt_1=[[3]], attempt_2=[[4]])
    assert challenges == original


def test_budget_expiry_still_saves_every_output_without_generation():
    challenges = {'a': dict(train=[], test=[dict(input=[[1]])])}
    saves = []
    def unexpected(*args, **kwargs):
        raise AssertionError('Generation after deadline')
    submission, events = run(challenges, None, 0, lambda s, e: saves.append(deepcopy(s)),
                             generator=unexpected, clock=lambda: 1)
    assert submission == saves[0]
    assert events == []
    validate_submission(challenges, submission)


def test_large_demonstration_outputs_receive_more_time_within_global_deadline():
    challenges = {
        'a': dict(train=[dict(input=[[0]], output=[[1]])], test=[dict(input=[[2]])]),
        'b': dict(train=[dict(input=[[0]*20]*20, output=[[1]*20]*20)], test=[dict(input=[[2]*20]*20)])}
    deadlines = []
    def generator(model, task, index, orientation, deadline):
        deadlines.append(deadline)
        return None, dict(reason='synthetic timeout')
    run(challenges, None, 200, lambda s, e: None, generator=generator, clock=lambda: 0)
    assert 0 < deadlines[0] < deadlines[1] <= 200
