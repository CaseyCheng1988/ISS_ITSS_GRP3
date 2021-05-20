from pyAudioAnalysis import MidTermFeatures as aF
import librosa

# extract mid term audio features across entire segment interval
def extractSegmentMtAudFt(aud_seg, sr, s_win_t=0.01, s_step_t=0.005):
    mid_window = len(aud_seg) - 1
    mid_step = mid_window
    return aF.mid_feature_extraction(aud_seg, sr, mid_window, mid_step,
                                     int(sr * s_win_t), int(sr * s_step_t))


# extract 10 mid-term features across an audio segment
def ExtractSegAudFt(seg_path):
    SF = 22050  # sampling rate
    M_WIN_TOTAL = 10  # total mid-term segments
    S_WIN_T = 0.01  # time length of short-term window
    S_STEP_T = 0.005  # time length of short-term hop

    seg_aud, _ = librosa.load(seg_path)
    m_win = int(len(seg_aud) / M_WIN_TOTAL)
    m_step = int(len(seg_aud) / M_WIN_TOTAL)
    s_win = int(SF * S_WIN_T)
    s_step = int(SF * S_STEP_T)

    mt, st, mt_n = aF.mid_feature_extraction(seg_aud, SF, m_win, m_step,
                                             s_win, s_step)

    if mt.shape[1] > M_WIN_TOTAL:
        return mt[:, :-1]
    else:
        return mt