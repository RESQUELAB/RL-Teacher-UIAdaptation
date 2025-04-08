# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import queue
import sys
import time
from datetime import datetime
from multiprocessing import Process, Queue, Value

import numpy as np

from ga3c.Config import Config

if sys.version_info >= (3, 0):
    from queue import Queue as queueQueue
else:
    from Queue import Queue as queueQueue


class ProcessStats(Process):
    def __init__(self):
        super(ProcessStats, self).__init__()
        self.episode_log_q = Queue(maxsize=100)
        self.episode_count = Value('i', 0)
        self.training_count = Value('i', 0)
        self.should_save_model = Value('i', 0)
        self.trainer_count = Value('i', 0)
        self.predictor_count = Value('i', 0)
        self.agent_count = Value('i', 0)
        self.total_frame_count = 0
        self.exit_flag = Value('i', 0)  # Exit flag to gracefully shut down the process

    def FPS(self):
        # average FPS from the beginning of the training (not current FPS)
        time_elapsed = time.time() - self.start_time
        # Avoid division by zero
        if time_elapsed <= 0:
            return 0  

        return np.ceil(self.total_frame_count / time_elapsed)

    def TPS(self):
        time_elapsed = time.time() - self.start_time
        # Avoid division by zero
        if time_elapsed <= 0:
            return 0  # or some other default value
        
        # average TPS from the beginning of the training (not current TPS)
        return np.ceil(self.training_count.value / time_elapsed)

    def run(self):
        with open(Config.RESULTS_FILENAME, 'a') as results_logger:
            rolling_frame_count = 0
            rolling_reward = 0
            results_q = queueQueue(maxsize=Config.STAT_ROLLING_MEAN_WINDOW)

            self.start_time = time.time()
            first_time = datetime.now()
            while True:
                # print("WHILE AGAIN: ", self.exit_flag, " - Val: " ,self.exit_flag.value)
                if self.exit_flag.value == 1:  # Check exit flag to stop process
                    print("Exiting ProcessStats as per exit_flag.")
                    break

                try:
                    episode_time, reward, length = self.episode_log_q.get(timeout=10)
                except queue.Empty:
                    print("No data in episode_log_q for 10 seconds")
                    continue

                results_logger.write('%s, %d, %d\n' % (episode_time.strftime("%Y-%m-%d %H:%M:%S"), reward, length))
                results_logger.flush()

                self.total_frame_count += length
                self.episode_count.value += 1

                rolling_frame_count += length
                rolling_reward += reward

                # COPYPASTA FROM Server.py TODO: Refactor! #
                step = min(self.episode_count.value, Config.ANNEALING_EPISODE_COUNT - 1)
                beta_multiplier = (Config.BETA_END - Config.BETA_START) / Config.ANNEALING_EPISODE_COUNT
                beta = Config.BETA_START + beta_multiplier * step
                ###

                if results_q.full():
                    old_episode_time, old_reward, old_length = results_q.get()
                    rolling_frame_count -= old_length
                    rolling_reward -= old_reward
                    first_time = old_episode_time

                results_q.put((episode_time, reward, length))
                if self.episode_count.value % Config.SAVE_FREQUENCY == 0:
                    print("self.episode_count:: " , self.episode_count)
                    self.should_save_model.value = 1

                if self.episode_count.value % Config.PRINT_STATS_FREQUENCY == 0:
                    
                    time_diff = (datetime.now() - first_time).total_seconds()

                    # Check if the time difference is too small or zero
                    if time_diff > 0:
                        pps = rolling_frame_count / time_diff
                    else:
                        pps = 0  # Or set it to a default safe value

                    print(
                        '[Time: %8d] '
                        '[Episode: %8d Score: %10.4f] '
                        '[RScore: %10.4f RPPS: %5d] '
                        '[PPS: %5d TPS: %5d] '
                        '[NT: %2d NP: %2d NA: %2d] '
                        '[Beta: %5.4f] '
                        % (int(time.time() - self.start_time),
                           self.episode_count.value, reward,
                           rolling_reward / results_q.qsize(),
                        #    rolling_frame_count / (datetime.now() - first_time).total_seconds(),
                           pps,
                           self.FPS(), self.TPS(),
                           self.trainer_count.value, self.predictor_count.value, self.agent_count.value,
                           beta))
                    sys.stdout.flush()
