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

from multiprocessing import Queue

import time
import gym

from ga3c.Config import Config
from ga3c.Environment import Environment
from ga3c.NetworkVP import NetworkVP
from ga3c.ProcessAgent import ProcessAgent
from ga3c.ProcessStats import ProcessStats
from ga3c.ThreadDynamicAdjustment import ThreadDynamicAdjustment
from ga3c.ThreadPredictor import ThreadPredictor
from ga3c.ThreadTrainer import ThreadTrainer


class Server:
    def __init__(self, reward_modifier=None):
        self.stats = ProcessStats()
        self.user = None
        self.domain = None

        if reward_modifier:
            self.reward_modifier_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)
            self.reward_modifier = reward_modifier
            self.user = reward_modifier.user
            self.domain = reward_modifier.domain
            

        self.training_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.prediction_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)

        self.model = NetworkVP(Config.DEVICE, 
                               Config.NETWORK_NAME, 
                               Environment().get_num_actions(),
                               user = self.user,
                               domain = self.domain)
        if Config.LOAD_CHECKPOINT:
            self.stats.episode_count.value = self.model.try_to_load()

        self.training_step = 0
        self.frame_counter = 0

        self.agents = []
        self.predictors = []
        self.trainers = []
        self.dynamic_adjustment = ThreadDynamicAdjustment(self)

    def add_agent(self):
        self.agents.append(ProcessAgent(
            len(self.agents), self.prediction_q, self.training_q, self.stats.episode_log_q, self.reward_modifier_q,
            env=Config.ENV, parentStats=self.stats))
        self.agents[-1].run()

    def remove_agent(self):
        if self.agents:
            agent = self.agents[-1]
            agent.exit_flag.value = True
            if agent.is_alive():
                agent.join()
            self.agents.pop()
        else:
            print("No agents to remove.")

    def add_predictor(self):
        self.predictors.append(ThreadPredictor(self, len(self.predictors)))
        self.predictors[-1].start()

    def remove_predictor(self):
        if self.predictors:
            predictor = self.predictors[-1]
            predictor.exit_flag = True
            if predictor.is_alive():
                print("joinning predictor")
                predictor.join()
                print("predictor joinned")
            self.predictors.pop()
        else:
            print("No predictors to remove.")

    def add_trainer(self):
        self.trainers.append(ThreadTrainer(self, len(self.trainers)))
        self.trainers[-1].start()

    def remove_trainer(self):
        if self.trainers:
            trainer = self.trainers[-1]
            trainer.exit_flag = True
            if trainer.is_alive():
                print("joinning trainner")
                trainer.join()
                print("trainer joinned")
            self.trainers.pop()
        else:
            print("No trainers to remove.")

    def remove_statsProces(self):
        if self.stats:
            self.stats.exit_flag.value = 1
            if self.stats.is_alive():
                print("joining the stats Process")
                self.stats.join()
                print("Stats Process joinned")
        else:
            print("No stats to remove.")

    def train_model(self, x_, r_, a_, trainer_id):
        self.model.train(x_, r_, a_, trainer_id)
        self.training_step += 1
        self.frame_counter += x_.shape[0]

        self.stats.training_count.value += 1
        self.dynamic_adjustment.temporal_training_count += 1

        if Config.TENSORBOARD and self.stats.training_count.value % Config.TENSORBOARD_UPDATE_FREQUENCY == 0:
            self.model.log(x_, r_, a_)

    def save_model(self):
        self.model.save(self.stats.episode_count.value)

    def main(self):
        gym.undo_logger_setup()

        self.stats.start()
        self.dynamic_adjustment.start()

        if Config.PLAY_MODE:
            print("\n\n\n")
            print("WE ARE IN PLAY MODE!!! NO TRAINNERS!!")
            print("\n\n\n")
            for trainer in self.trainers:
                trainer.enabled = False
        else:
            print("\n\n\n")
            print("WE ARE NOT IN PLAY MODE!!! GETTING DA TRAINNERS!!")
            print(Config.SAVE_MODELS)
            print("\n\n\n")
            
        learning_rate_multiplier = (Config.LEARNING_RATE_END - Config.LEARNING_RATE_START) / Config.ANNEALING_EPISODE_COUNT
        beta_multiplier = (Config.BETA_END - Config.BETA_START) / Config.ANNEALING_EPISODE_COUNT
        while self.stats.episode_count.value < Config.EPISODES:
            step = min(self.stats.episode_count.value, Config.ANNEALING_EPISODE_COUNT - 1)
            self.model.learning_rate = Config.LEARNING_RATE_START + learning_rate_multiplier * step
            self.model.beta = Config.BETA_START + beta_multiplier * step

            # Saving is async - even if we start saving at a given episode, we may save the model at a later episode
            if Config.SAVE_MODELS and self.stats.should_save_model.value > 0:
                print("Saving GA3C model!")
                self.save_model()
                self.stats.should_save_model.value = 0

            if self.reward_modifier:
                ################################
                #  START REWARD MODIFICATIONS  #
                ################################
                if not self.reward_modifier_q.empty():
                    source_id, done, path = self.reward_modifier_q.get()
                    rewards = self.reward_modifier.predict_reward(path)

                    # if done:
                    #     self.reward_modifier.path_callback(path)

                    self.agents[source_id].wait_q.put(rewards)
                ################################
                #   END REWARD MODIFICATIONS   #
                ################################

            time.sleep(0.01)

        self.dynamic_adjustment.exit_flag = True
        while self.agents:
            self.remove_agent()
        while self.predictors:
            self.remove_predictor()
        while self.trainers:
            self.remove_trainer()
        self.remove_statsProces()
        print("Finished the Server MAIN.")
