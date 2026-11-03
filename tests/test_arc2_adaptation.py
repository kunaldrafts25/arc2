import random

from arc2.adaptation import augmented_pair,training_tokens


class CharacterTokenizer:
    def encode(self,text,**kwargs):return list(map(ord,text))


def test_adaptation_masks_input_and_header_and_trains_only_answer_suffix():
    pair=dict(input=[[1,2]],output=[[3,4]])
    ids,labels=training_tokens(pair,CharacterTokenizer())
    start=next(i for i,label in enumerate(labels) if label!=-100)
    assert ''.join(map(chr,ids[start:]))=='34<|im_end|>'
    assert '12' in ''.join(map(chr,ids[:start]))
    assert all(x==-100 for x in labels[:start])


def test_augmentation_applies_one_bijection_and_orientation_to_both_sides():
    pair=dict(input=[[1,2],[3,4]],output=[[1,2],[3,4]])
    result=augmented_pair(pair,random.Random(42))
    assert result['input']==result['output']
    assert len(set(c for row in result['input'] for c in row))==4
    assert pair['input']==[[1,2],[3,4]]
