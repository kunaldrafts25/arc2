import pytest

from arc2.neural import allowed_tokens,parse_output,prompt


def test_prompt_uses_only_demonstrations_and_selected_test_input():
    task=dict(train=[dict(input=[[1,2]],output=[[3]])],test=[dict(input=[[4]],output=[[9]])])
    text=prompt(task,0)
    assert text=='<|im_start|>user\n12<|im_end|><|im_start|>assistant\n3<|im_end|><|im_start|>user\n4<|im_end|><|im_start|>assistant\n'
    assert '9' not in text


def test_parser_rejects_non_grid_text_ragged_rows_and_invalid_dimensions():
    assert parse_output('12\n34<|im_end|>')==((1,2),(3,4))
    for text in ('1\n23','answer: 12','12 34','1\n\n1','1'*31):
        with pytest.raises(ValueError):parse_output(text)


def test_generation_grammar_requires_nonempty_rectangular_rows_and_caps_size():
    digits=list(range(10));nl=10;eos=15
    assert allowed_tokens([],digits,nl,eos)==digits
    assert eos in allowed_tokens([1,2],digits,nl,eos)
    assert allowed_tokens([1,2,nl,3],digits,nl,eos)==digits
    assert allowed_tokens([1,2,nl,3,4],digits,nl,eos)==[eos,nl]
    assert allowed_tokens([1]*30,digits,nl,eos)==[nl,eos]
    assert allowed_tokens(([1,nl]*29)+[1],digits,nl,eos)==[eos]


def test_transposed_prompt_preserves_coordinate_convention():
    task=dict(train=[],test=[dict(input=[[1,2,3],[4,5,6]])])
    assert '14\n25\n36' in prompt(task,0,orientation=4)
