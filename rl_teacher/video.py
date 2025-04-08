import distutils.spawn
import distutils.version
import os
import os.path as osp
import subprocess

import numpy as np
from gym import error

from PIL import Image

def upload_to_gcs(local_path, gcs_path):
    assert osp.isfile(local_path), "%s must be a file" % local_path
    assert gcs_path.startswith("gs://"), "%s must start with gs://" % gcs_path

    print("Copying media to %s in a background process" % gcs_path)
    subprocess.check_call(['gsutil', 'cp', local_path, gcs_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    

def write_segment_to_video(frames, fname, fps):
    os.makedirs(osp.dirname(fname), exist_ok=True)
    # Draw out the last frame by 0.2s
    for i in range(int(fps * 0.2)):
        frames.append(frames[-1])
    export_video(frames, fname, fps=fps)

def export_video(frames, fname, fps=10):
    assert "mp4" in fname, "Name requires .mp4 suffix"
    assert osp.isdir(osp.dirname(fname)), "%s must be a directory" % osp.dirname(fname)

    # Resize frames to have dimensions divisible by 2
    frames_resized = []
    for frame in frames:
        if isinstance(frame, tuple):
            frames_resized.append(frame)  # Raw image frames, no need to resize
        else:
            if isinstance(frame, np.ndarray):
                w, h = frame.shape[:2]
                w = w - (w % 2)  # Make width divisible by 2
                h = h - (h % 2)  # Make height divisible by 2
                frame_resized = frame[:w, :h]
                frames_resized.append(frame_resized)

    '''
        MAKE SURE IF THE FRAME IS IN A VIDEO FORMAT OR IN UI DESIGN FORMAT.
        THEN CREATE THE CORRESPONDING VIDEO.
    '''
    print("EXPORT VIDEO!")
    raw_image = isinstance(frames_resized[0], tuple)
    shape = frames_resized[0][0] if raw_image else frames_resized[0].shape
    greyscale = (len(shape) == 2)
    if greyscale: 
        shape = shape + (3,)
    encoder = ImageEncoder(fname, shape, fps)
    for frame in frames_resized:
        if raw_image:
            print("raw image!")
            encoder.proc.stdin.write(frame[1])
        else:
            if greyscale:
                # Convert cells of domain [-1, 1) to triplets of range [0, 256)
                frame = np.transpose(np.tile((frame + 1) * 128, (3, 1, 1)), (1, 2, 0)).astype(np.uint8)
            else:
                if frame.shape[-1] == 4:  # BGRA to RGBA
                    frame = frame[..., [2, 1, 0, 3]]  # Swap B and R channels, keep G and A
                    
            encoder.capture_frame(frame)
    encoder.close()

class SegmentVideoRecorder(object):
    """ Wraps a reward model's path_callback to regularly save a video to disk every so often. """

    def __init__(self, model, env, save_dir, checkpoint_interval=500):
        self.model = model
        self.env = env
        self.checkpoint_interval = checkpoint_interval
        self.save_dir = save_dir

        self._num_paths_seen = 0  # Internal counter of how many paths we've seen

    def path_callback(self, path):
        if self._num_paths_seen % self.checkpoint_interval == 0:  # and self._num_paths_seen != 0:
            fname = '%s/run_%s.mp4' % (self.save_dir, self._num_paths_seen)
            print("Saving video of run %s to %s" % (self._num_paths_seen, fname))
            frames = [self.env.render_full_obs(x) for x in path["human_obs"]]
            write_segment_to_video(frames, fname, self.env.fps)
        self._num_paths_seen += 1

        self.model.path_callback(path)

    def predict_reward(self, path):
        return self.model.predict_reward(path)

class ImageEncoder(object):
    """ Wrapper around a video encoding, command-line utility. """

    def __init__(self, output_path, frame_shape, frames_per_sec):
        self.proc = None
        self.output_path = output_path
        # Frame shape should be lines-first, so w and h are swapped
        h, w, pixfmt = frame_shape
        if pixfmt != 3 and pixfmt != 4:
            raise Exception(
                "Your frame has shape {}, but we require (w,h,3) or (w,h,4), i.e. RGB values for a w-by-h image, with an optional alpha channl.".format(
                    frame_shape))
        self.wh = (w, h)
        self.includes_alpha = (pixfmt == 4)
        self.frame_shape = frame_shape
        self.frames_per_sec = frames_per_sec

        if distutils.spawn.find_executable('avconv') is not None:
            self.backend = 'avconv'
        elif distutils.spawn.find_executable('ffmpeg') is not None:
            self.backend = 'ffmpeg'
        else:
            raise Exception(
                """Found neither the ffmpeg nor avconv executables. On OS X, you can install ffmpeg via `brew install ffmpeg`. On most Ubuntu variants, `sudo apt-get install ffmpeg` should do it. On Ubuntu 14.04, however, you'll need to install avconv with `sudo apt-get install libav-tools`.""")

        self.start()

    @property
    def version_info(self):
        return {
            'backend': self.backend,
            'version': str(subprocess.check_output(
                [self.backend, '-version'],
                stderr=subprocess.STDOUT)),
            'cmdline': self.cmdline
        }

    def start(self):
        self.cmdline = (
            self.backend,
            '-nostats',
            '-loglevel', 'error',  # suppress warnings
            '-y',
            '-r', '%d' % self.frames_per_sec,
            '-f', 'rawvideo',  # input
            '-s:v', '{}x{}'.format(*self.wh),
            '-pix_fmt', ('rgb32' if self.includes_alpha else 'rgb24'),
            '-i', '-',  # this used to be /dev/stdin, which is not Windows-friendly
            '-vf', 'vflip',
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            self.output_path
        )

        if hasattr(os, 'setsid'):  # setsid not present on Windows
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE, preexec_fn=os.setsid)
        else:
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE)

    def capture_frame(self, frame):
        if not isinstance(frame, (np.ndarray, np.generic)):
            raise error.InvalidFrame(
                'Wrong type {} for {} (must be np.ndarray or np.generic)'.format(type(frame), frame))
        if frame.shape != self.frame_shape:
            raise error.InvalidFrame(
                "Your frame has shape {}, but the VideoRecorder is configured for shape {}.".format(
                    frame.shape, self.frame_shape))
        if frame.dtype != np.uint8:
            raise error.InvalidFrame(
                "Your frame has data type {}, but we require uint8 (i.e. RGB values from 0-255).".format(frame.dtype))

        if distutils.version.LooseVersion(np.__version__) >= distutils.version.LooseVersion('1.9.0'):
            self.proc.stdin.write(frame.tobytes())
        else:
            self.proc.stdin.write(frame.tostring())

    def close(self):
        self.proc.stdin.close()
        ret = self.proc.wait()
        if ret != 0:
            raise Exception("VideoRecorder encoder exited with status {}".format(ret))
