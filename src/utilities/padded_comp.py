import numpy as np

def padded_comp(c1, c2, padding_mask): 
    """Takes two numpy arrays to compare and then a padding mask of the same shape, masking out some elements from comparison"""
    assert c1.shape == c2.shape, "Inputs must have same shape!"
    assert padding_mask.shape == c1.shape[0:2], "Padding mask must be same shape as inputs!"
    res = []
    for i in range(padding_mask.shape[0]): 
        # for each input sequence
        res.append(np.allclose(c1[i][padding_mask[i]], c2[i][padding_mask[i]]))
    return(res)

