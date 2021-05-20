import numpy as np
from videoprops import get_video_properties
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# get average fps from file metadata
def get_vid_fps(vid):
  fps = get_video_properties(vid)['avg_frame_rate']
  fps = fps.split('/')
  fps = float(fps[0])/float(fps[1])
  return fps

def get_subclip(vid, dst_vid, t_int):
  ffmpeg_extract_subclip(vid, t_int[0], t_int[1], targetname=dst_vid)

# convert frame intervals to time interval given frame rate
def frameToTime(frame_intervals, fps):
  ret = np.zeros((len(frame_intervals), 2))
  for i, interval in enumerate(frame_intervals):
    ret[i,:] = np.array([interval[0]/fps, interval[1]/fps])
  return ret
