import numpy as np

# pad audio segment
def padAudio(aud, sr, num_front_pad, num_back_pad, mode='zero'):
    dtype = aud.dtype

    if mode == 'zero':
        front_pad = np.zeros((num_front_pad,), dtype=dtype)
        back_pad = np.zeros((num_back_pad,), dtype=dtype)
    elif mode == 'mirror':
        front_pad = np.flip(aud[1:1 + num_front_pad])
        back_pad = np.flip(aud[-1 - num_back_pad:-1])

    return np.concatenate((front_pad, aud, back_pad), axis=0)

# get new index of padded audio samples
def paddedIdx(anchors, max_front_pad):
    ret = anchors + max_front_pad
    return ret.astype(np.int)