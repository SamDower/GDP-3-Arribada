r"""
Module to run an animal detection model on lots of images, writing the results
to a file in the same format produced by our batch API:

https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing

This enables the results to be used in our post-processing pipeline; see
api/batch_processing/postprocessing/postprocess_batch_results.py .

This script can save results to checkpoints intermittently, in case disaster
strikes. To enable this, set --checkpoint_frequency to n > 0, and results 
will be saved as a checkpoint every n images. Checkpoints will be written 
to a file in the same directory as the output_file, and after all images
are processed and final results file written to output_file, the temporary
checkpoint file will be deleted. If you want to resume from a checkpoint, set
the checkpoint file's path using --resume_from_checkpoint.

The `threshold` you can provide as an argument is the confidence threshold above
which detections will be included in the output file.

Has preliminary multiprocessing support for CPUs only; if a GPU is available, it will
use the GPU instead of CPUs, and the --ncores option will be ignored.  Checkpointing
is not supported when using multiprocessing.

Does not have a command-line option to bind the process to a particular GPU, but you can 
prepend with "CUDA_VISIBLE_DEVICES=0 ", for example, to bind to GPU 0, e.g.:

CUDA_VISIBLE_DEVICES=0 python detection/run_detector_batch.py ~/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb ~/data/test-small ~/tmp/mdv4test.json --output_relative_filenames --recursive

"""

#%% Constants, imports, environment

import argparse
import json
import os
import sys
import time
import copy
import shutil
import warnings
import itertools

from datetime import datetime
from functools import partial

import humanfriendly
from tqdm import tqdm

# from multiprocessing.pool import ThreadPool as workerpool
import multiprocessing
from threading import Thread
from multiprocessing import Process
from multiprocessing.pool import Pool as workerpool

# Number of images to pre-fetch
max_queue_size = 10
use_threads_for_queue = False
verbose = False

# Useful hack to force CPU inference.
#
# Need to do this before any PT/TF imports, which happen when we import
# from run_detector.
force_cpu = False
if force_cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
import subprocess
from PIL import Image

# Add Python paths
dirname = os.path.dirname(__file__)
sys.path.append(os.path.join(dirname, 'cameratraps'))
sys.path.append(os.path.join(dirname, 'ai4eutils'))
sys.path.append(os.path.join(dirname, 'yolov5'))

# PyTorch fix
# https://github.com/ultralytics/yolov5/issues/6948
import torch
import torch.nn as nn
import torch.nn.functional as F

def forward(self, input: torch.Tensor) -> torch.Tensor:
    return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)
nn.Upsample.forward = forward

from detection.run_detector import ImagePathUtils, is_gpu_available,\
    load_detector,\
    get_detector_version_from_filename,\
    get_detector_metadata_from_version_string,\
    FAILURE_INFER, FAILURE_IMAGE_OPEN,\
    DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD, DEFAULT_DETECTOR_LABEL_MAP,\
    DEFAULT_BOX_THICKNESS, DEFAULT_BOX_EXPANSION

import visualization.visualization_utils as viz_utils

# Numpy FutureWarnings from tensorflow import
warnings.filterwarnings('ignore', category=FutureWarning)


#%% Support functions for multiprocessing

def producer_func(in_queue, stream_link):
    """ 
    Producer function for a live stream.

    in_queue: queue to enqueue to
    stream_link: URL of the stream (stream_link = 0 for webcam).

    Reads up to N images from stream and puts them on the blocking queue for processing.
    """
    if verbose:
        print('[Input thread] Starting', flush=True)

    num_frames = 0
    cap = cv2.VideoCapture(stream_link)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('[Input thread] Error retrieving frame from stream (ret = False)')
            continue

        try:
            if verbose:
                print('[Input thread] Loading frame {}'.format(num_frames), flush=True)
            image = array_to_image(frame)
        except Exception as e:
            print('[Input thread] Frame {} cannot be loaded. Exception: {}'.format(frame, e))
            # raise
            continue

        if verbose:
            print('[Input thread] Queueing frame {}'.format(num_frames), flush=True)
        num_frames += 1

        in_queue.put(image)
    
    cap.release()
    in_queue.put(None)
        
    print('[Input thread] Exiting', flush=True)
    
    
def consumer_func(in_queue: multiprocessing.JoinableQueue,
                  out_queue: multiprocessing.JoinableQueue,
                  model_file, confidence_threshold):
    """ 
    Consumer function
    
    Pulls images from a blocking queue and processes them.
    """
    
    # Record time
    if verbose:
        print('[Main thread] Starting', flush=True)
    start_time = time.time()
    detector = load_detector(model_file)
    elapsed = time.time() - start_time
    print('[Main thread] Loaded model (before queueing) in {}'.format(humanfriendly.format_timespan(elapsed)), flush=True)
    num_frames = 0
    
    while True:
        image = in_queue.get()
        if image is None:
            in_queue.task_done()
            return

        # Run model to get results
        result = detector.generate_detections_one_image(
            image, 'frame_{}'.format(num_frames), detection_threshold=confidence_threshold)

        # Enqueue results
        out_queue.put((image, result))
        in_queue.task_done()

        # Record time
        if verbose:
            print('[Main thread] Processed image {}'.format(num_frames), flush=True)
        num_frames += 1


def output_func(out_queue, stream_link, confidence_threshold, fps,
                box_thickness=DEFAULT_BOX_THICKNESS,
                box_expansion=DEFAULT_BOX_EXPANSION):
    """ 
    Streams the results to a URL.

    out_queue: queue to dequeue from
    stream_link: URL of the stream
    """
    if verbose:
        print('[Output thread] Starting', flush=True)

    start_time = time.time()
    num_frames = 0
    num_frames_since = 0
    ffmpeg_process = None

    while True:
        # Dequeue detection result
        r = out_queue.get()
        if r is None:
            out_queue.task_done()
            return
        
        image, result = r

        # Start ffmpeg if not started
        if ffmpeg_process is None:
            try:
                command = [
                    'ffmpeg',
                    '-v', 'verbose',
                    '-y',
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-pix_fmt', 'bgr24',
                    '-s', "{}x{}".format(image.width, image.height),
                    '-r', str(fps),
                    '-i', '-',
                    # '-c:v', 'libx264',
                    # '-pix_fmt', 'yuv420p',
                    # '-preset', 'ultrafast',
                    # '-f', 'flv',
                    '-c:v', 'libx264',
                    '-f', 'mpegts', # MPEG encoding
                    stream_link]
                print('[Output thread] Starting ffmpeg: {}'.format(' '.join((command))), flush=True)
                ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE)
            except Exception as e:
                print('[Output thread] Unable to start ffmpeg: ')
                raise

        # Draw bounding boxes (in-place)
        viz_utils.render_detection_bounding_boxes(
            result['detections'], image,
            label_map=DEFAULT_DETECTOR_LABEL_MAP,
            confidence_threshold=confidence_threshold,
            thickness=box_thickness, expansion=box_expansion)

        try:
            # image.show()
            ffmpeg_process.stdin.write(image.tobytes())
        except Exception as e:
            print('[Output thread] Frame cannot be written to ffmpeg.')
            raise

        out_queue.task_done()

        # Record time
        if verbose:
            print('[Output thread] Finished frame {}'.format(num_frames), flush=True)
        num_frames += 1
        num_frames_since += 1
        if verbose or ((num_frames_since % 10) == 0):
            elapsed = time.time() - start_time
            images_per_second = num_frames_since / elapsed
            print('[Output thread] De-queued image {} ({:.2f}/s)'.format(
                num_frames, images_per_second), flush=True)
            # Changed to record current speed
            start_time = time.time()
            num_frames_since = 0
            

def run_detector_with_image_queue(in_stream_link, out_stream_link,
                                  model_file, confidence_threshold,
                                  fps, num_threads):
    """
    Driver function for the multiprocessing-based image queue.
    Starts a reader process to read images from stream, but 
    processes images in the  process from which this function is called (i.e., does not currently
    spawn a separate consumer process).
    """

    if in_stream_link.isdecimal():
        in_stream_link = int(in_stream_link)

    print('GPU available: {}'.format(is_gpu_available(model_file)))
    
    in_queue = multiprocessing.JoinableQueue(max_queue_size)
    out_queue = multiprocessing.JoinableQueue(max_queue_size)
    
    thread_class = Thread if use_threads_for_queue else Process
    input_thread = thread_class(target=producer_func, args=(in_queue, in_stream_link,))
    input_thread.daemon = False
    input_thread.start()
    
    output_thread = thread_class(target=output_func, args=(out_queue, out_stream_link,
                                                           confidence_threshold, fps))
    output_thread.daemon = True
    output_thread.start()
 
    # TODO
    #
    # The queue system is a little more elegant if we start one thread for reading and one
    # for processing, and this works fine on Windows, but because we import TF at module load,
    # CUDA will only work in the main process, so currently the consumer function runs here.
    #
    # To enable proper multi-GPU support, we may need to move the TF import to a separate module
    # that isn't loaded until very close to where inference actually happens.

    consumers = []
    if num_threads > 0:
        for _ in range(num_threads):
            consumer = thread_class(target=consumer_func,
                                    args=(in_queue, out_queue, model_file,
                                          confidence_threshold,))
            consumer.daemon = True
            consumer.start()
            consumers.append(consumer)
    else:
        consumer_func(in_queue,out_queue,model_file,confidence_threshold)

    input_thread.join()
    print('Producer finished')
   
    for consumer in consumers:
        consumer.join()
        print('Consumer finished')
    
    output_thread.join()
    print('Output thread finished')
    
    in_queue.join()
    out_queue.join()
    print('Queues joined')


# largely copied from visualization_utils.py
def array_to_image(array):
    image = Image.fromarray(array)
    if image.mode not in ('RGBA', 'RGB', 'L', 'I;16'):
        raise AttributeError(
            f'Image uses unsupported mode {image.mode}')
    if image.mode == 'RGBA' or image.mode == 'L':
        # PIL.Image.convert() returns a converted copy of this image
        image = image.convert(mode='RGB')

    # Alter orientation as needed according to EXIF tag 0x112 (274) for Orientation
    #
    # https://gist.github.com/dangtrinhnt/a577ece4cbe5364aad28
    # https://www.media.mit.edu/pia/Research/deepview/exif.html
    #
    IMAGE_ROTATIONS = {
        3: 180,
        6: 270,
        8: 90
    }
    try:
        exif = image._getexif()
        orientation: int = exif.get(274, None)  # 274 is the key for the Orientation field
        if orientation is not None and orientation in IMAGE_ROTATIONS:
            image = image.rotate(IMAGE_ROTATIONS[orientation], expand=True)  # returns a rotated copy
    except Exception:
        pass

    return image

    
#%% Command-line driver

def main():
    
    parser = argparse.ArgumentParser(
        description='Module to run a TF/PT animal detection model on a live stream')
    parser.add_argument(
        'detector_file',
        help='Path to detector model file (.pb or .pt)')
    parser.add_argument(
        'in_stream_link',
        type=str,
        help='Input stream link (or camera #)')
    parser.add_argument(
        'out_stream_link',
        type=str,
        help='Output stream link')
    parser.add_argument(
        '--threshold',
        type=float,
        default=DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD,
        help="Confidence threshold between 0 and 1.0, don't include boxes below this " + \
            "confidence in the output file. Default is {}".format(
                DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD))
    parser.add_argument(
        '--fps',
        type=float,
        default=2,
        help='Maximum frames per second of the output stream. '
             'Should be greater, but not much more than, the actual inference speed')
    parser.add_argument(
        '--num_threads',
        type=int,
        default=-1,
        help='Number of threads to use for inference')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    assert os.path.exists(args.detector_file), \
        'detector file {} does not exist'.format(args.detector_file)
    assert 0.0 < args.threshold <= 1.0, 'Confidence threshold needs to be between 0 and 1'
    assert args.num_threads != 0, 'num_threads must be -1 or positive'

    run_detector_with_image_queue(
        in_stream_link=args.in_stream_link,
        out_stream_link=args.out_stream_link,
        model_file=args.detector_file, 
        confidence_threshold=args.threshold,
        fps=args.fps,
        num_threads=args.num_threads)

    print('Done!')


if __name__ == '__main__':
    main()
