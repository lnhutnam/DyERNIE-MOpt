import random
import numpy as np


# #################Negative Sampling Methods#################################################
# generate negative samples by corrupting head or tail with equal probabilities with checking whether false negative samples exist.
def getBatch_filter_all_dyn(
    quadruples, entityTotal, srt_vocab, ort_vocab=None, corrupt_head=False, mult_num=1
):
    """
    quadruples: training batch
    entityTotal: the number of entities in the whole dataset
    corrupt_head: whether to corrupt the subject entity
    mult_num: number of negative samples
    """
    newQuadrupleList = [
        (
            corrupt_head_filter_dyn(quadruples[i], entityTotal, ort_vocab)
            if corrupt_head
            else corrupt_tail_filter_dyn(quadruples[i], entityTotal, srt_vocab)
        )
        for i in range(len(quadruples))
    ]
    batch_list = []
    batch_list.append(np.array(newQuadrupleList))
    if mult_num > 1:
        for i in range(0, mult_num - 1):
            newQuadrupleList2 = [
                (
                    corrupt_head_filter_dyn(quadruples[i], entityTotal, ort_vocab)
                    if corrupt_head
                    else corrupt_tail_filter_dyn(quadruples[i], entityTotal, srt_vocab)
                )
                for i in range(len(quadruples))
            ]
            batch_list.append(np.array(newQuadrupleList2))
        batch_list = np.stack(batch_list, axis=1)  # shape: batch_size * self.nneg * 4
    return batch_list


def getBatch_filter_all_static(
    quadruples, entityTotal, sr_vocab, or_vocab=None, corrupt_head=False, mult_num=1
):
    """
    quadruples: training batch
    entityTotal: the number of entities in the whole dataset
    corrupt_head: whether to corrupt the subject entity
    mult_num: number of negative samples
    """
    newQuadrupleList = [
        (
            corrupt_head_filter_static(quadruples[i], entityTotal, or_vocab)
            if corrupt_head
            else corrupt_tail_filter_static(quadruples[i], entityTotal, sr_vocab)
        )
        for i in range(len(quadruples))
    ]
    batch_list = []
    batch_list.append(np.array(newQuadrupleList))
    if mult_num > 1:
        for i in range(0, mult_num - 1):
            newQuadrupleList2 = [
                (
                    corrupt_head_filter_static(quadruples[i], entityTotal, or_vocab)
                    if corrupt_head
                    else corrupt_tail_filter_static(
                        quadruples[i], entityTotal, sr_vocab
                    )
                )
                for i in range(len(quadruples))
            ]
            batch_list.append(np.array(newQuadrupleList2))
        batch_list = np.stack(batch_list, axis=1)  # shape: batch_size * self.nneg * 4
    return batch_list


# Change the tail of a triple randomly,
# with checking whether it is a false negative sample.
# If it is, regenerate.
def corrupt_tail_filter_dyn(quadruple, entityTotal, quadrupleDict):
    while True:
        newTail = random.randrange(entityTotal)
        if newTail not in set(
            quadrupleDict[(quadruple[0], quadruple[1]), quadruple[3]]
        ):
            break
    return [quadruple[0], quadruple[1], newTail, quadruple[3]]


# Change the head of a triple randomly,
# with checking whether it is a false negative sample.
# If it is, regenerate.
def corrupt_head_filter_dyn(quadruple, entityTotal, quadrupleDict):
    while True:
        newHead = random.randrange(entityTotal)
        if newHead not in set(
            quadrupleDict[(quadruple[2], quadruple[1], quadruple[3])]
        ):
            break
    return [newHead, quadruple[1], quadruple[2], quadruple[3]]


# If it is, regenerate.
def corrupt_tail_filter_static(quadruple, entityTotal, tripleDict):
    while True:
        newTail = random.randrange(entityTotal)
        if newTail not in set(tripleDict[(quadruple[0], quadruple[1])]):
            break
    return [quadruple[0], quadruple[1], newTail, quadruple[3]]


# Change the head of a triple randomly,
# with checking whether it is a false negative sample.
# If it is, regenerate.
def corrupt_head_filter_static(quadruple, entityTotal, tripleDict):
    while True:
        newHead = random.randrange(entityTotal)
        if newHead not in set(tripleDict[(quadruple[2], quadruple[1])]):
            break
    return [newHead, quadruple[1], quadruple[2], quadruple[3]]
