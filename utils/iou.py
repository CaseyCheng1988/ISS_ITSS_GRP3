import numpy as np

# check if two intervals overlap
def isOverlap(src, ref):
  if src[1] <= ref[0] or src[0] >= ref[1]:
    return False
  else:
    return True

# find overlapping interval
def findOverlap(src, ref):
  dtype = src.dtype
  if not isOverlap(src, ref):
    return np.array([None, None])
  else:
    if src[0] <= ref[0]:
      if src[1] <= ref[1]:
        return np.array([ref[0], src[1]], dtype=dtype)
      else:
        return np.array([ref[0], ref[1]], dtype=dtype)
    else:
      if src[1] <= ref[1]:
        return np.array([src[0], src[1]], dtype=dtype)
      else:
        return np.array([src[0], ref[1]], dtype=dtype)

# compute intersection over union
def findIou(src, ref):
  if not isOverlap(src, ref):
    return 0.0
  else:
    overlap = findOverlap(src, ref)
    overlap = overlap[1] - overlap[0]

    union = max(np.max(src), np.max(ref)) - min(np.min(src), np.min(ref))

    return overlap/float(union)

# compute overlaps of source interval with reference intervals in an array
def findOverlapWithRefArr(src, ref_arr):
  ret = np.zeros_like(ref_arr)
  for i, ref in enumerate(ref_arr):
    ret[i, :] = findOverlap(src, ref)
  return ret

# compute IoU of source interval with reference intervals in an array
def findIouWithRefArr(src, ref_arr):
  ret = np.zeros(len(ref_arr), dtype=np.float32)
  for i, ref in enumerate(ref_arr):
    ret[i] = findIou(src,ref)
  return ret

# remove overlapping intervals using non-maximal suppression
def nms(confidence, location, thresh):
    sort_idx = np.flip(np.argsort(confidence)) # get sort permutation
    confidence= np.flip(np.sort(confidence)) # sort in decreasing order
    location = location[sort_idx] # organize location according to sort order

    final_proposal = []
    final_score = []
    proposal = list(location)
    score = list(confidence)
    while len(proposal) > 0:
        # get topmost proposal for downstream comparison
        top_proposal = proposal[0]

        # move top proposal to final proposal
        final_proposal.append(proposal[0])
        final_score.append(score[0])

        # remove top proposal from the list
        proposal = proposal[1:]
        score = score[1:]

        # break out if nothing left in proposal
        if len(proposal) == 0:
            break

        next_proposal = []
        next_score = []
        for i, i_prop in enumerate(proposal):
            iou = findIou(top_proposal, i_prop)
            if iou < thresh: # only retain non-overlapping intervals
                next_proposal.append(i_prop)
                next_score.append(score[i])

        proposal = next_proposal
        score = next_score

    return final_proposal, final_score