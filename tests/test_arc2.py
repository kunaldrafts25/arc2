from copy import deepcopy

import pytest

from arc2.symbolic import grid,solve,orient
from arc2.evaluation import validate_submission,score_submission


def test_rotation_and_color_rule_generalize_from_demonstrations():
    def rule(g):return [[{0:0,1:3,2:4}[c] for c in row] for row in orient(grid(g),1)]
    xs=[[[1,0,2],[2,1,0]],[[2,0],[1,2],[0,1]]]
    test=[[2,1,0],[0,2,1],[1,0,2]]
    task=dict(train=[dict(input=x,output=rule(x)) for x in xs],test=[dict(input=test)])
    before=deepcopy(task);attempts,info=solve(task,seconds=2)
    assert rule(test) in attempts[0].values() and info['consistent_programs']>0
    assert task==before


def test_symmetric_examples_do_not_erase_distinct_test_hypotheses():
    task=dict(train=[dict(input=[[1,2],[2,1]],output=[[1,2],[2,1]])],
        test=[dict(input=[[1,0],[3,2]])])
    attempts,info=solve(task,seconds=2)
    assert info['consistent_programs']>1
    assert attempts[0]['attempt_1']!=attempts[0]['attempt_2']


def test_crop_discovers_a_rule_without_task_id_or_test_labels():
    task=dict(train=[dict(input=[[0,0,0],[0,2,0],[0,0,0]],output=[[2]]),
        dict(input=[[0,0,0,0],[0,3,3,0],[0,3,3,0]],output=[[3,3],[3,3]])],
        test=[dict(input=[[0,0,0,0],[0,1,2,0],[0,4,5,0]])])
    attempts,_=solve(task,seconds=2)
    assert [[1,2],[4,5]] in attempts[0].values()


def test_metric_weights_test_outputs_not_tasks_and_accepts_either_attempt():
    challenges={'a':dict(test=[dict(input=[[0]]),dict(input=[[0]])]),'b':dict(test=[dict(input=[[0]])])}
    submission={'a':[dict(attempt_1=[[0]],attempt_2=[[2]]),dict(attempt_1=[[0]],attempt_2=[[0]])],
                'b':[dict(attempt_1=[[0]],attempt_2=[[0]])]}
    result=score_submission(challenges,dict(a=[[[2]],[[3]]],b=[[[4]]]),submission)
    assert result['correct_outputs']==1 and result['test_outputs']==3
    assert result['score']==pytest.approx(100/3)


def test_submission_requires_every_test_output_and_valid_grid_values():
    challenges={'a':dict(test=[dict(input=[[0]]),dict(input=[[0]])])}
    with pytest.raises(ValueError):validate_submission(challenges,{'a':[dict(attempt_1=[[0]],attempt_2=[[0]])]})
    with pytest.raises(ValueError):grid([[True]])
    with pytest.raises(ValueError):grid([[10]])
    with pytest.raises(ValueError):grid([[0],[0,1]])
