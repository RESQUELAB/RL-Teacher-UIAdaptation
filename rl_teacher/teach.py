import os
import argparse
import multiprocessing
from time import time

import numpy as np
import tensorflow as tf

from rl_teacher.reward_models import OriginalEnvironmentReward, OrdinalRewardModel
from rl_teacher.envs import make_env
from rl_teacher.label_schedules import make_label_schedule
from rl_teacher.video import SegmentVideoRecorder
from rl_teacher.episode_logger import EpisodeLogger
from rl_teacher.utils import slugify
from rl_teacher.utils import get_timesteps_per_episode

def main():
    # Tensorflow is not fork-safe, so we must use spawn instead
    # https://github.com/tensorflow/tensorflow/issues/5448#issuecomment-258934405
    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env_id', required=True)
    parser.add_argument('-p', '--reward_model', required=True)
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-s', '--seed', default=1, type=int)
    parser.add_argument('-w', '--workers', default=4, type=int)
    parser.add_argument('-l', '--n_labels', default=None, type=int)
    parser.add_argument('-L', '--pretrain_labels', default=None, type=int)
    parser.add_argument('-t', '--num_timesteps', default=5e6, type=int)
    parser.add_argument('-a', '--agent', default="ga3c", type=str)
    parser.add_argument('-i', '--pretrain_iters', default=5000, type=int)
    parser.add_argument('-b', '--starting_beta', default=0.1, type=float)
    parser.add_argument('-c', '--clip_length', default=1.5, type=float)
    parser.add_argument('-f', '--stacked_frames', default=4, type=int)
    parser.add_argument('-V', '--no_videos', action="store_true")
    parser.add_argument('--force_new_environment_clips', action="store_true")
    parser.add_argument('--force_new_training_labels', action="store_true")
    parser.add_argument('--force_new_reward_model', action="store_true")
    parser.add_argument('--force_new_agent_model', action="store_true")
    args = parser.parse_args()

    env_id = args.env_id
    experiment_name = slugify(args.name)

    # Potentially erase old data
    if args.force_new_environment_clips:
        existing_clips = [x for x in os.listdir('clips') if x.startswith(env_id)]
        if len(existing_clips):
            print("Found {} old clips".format(len(existing_clips)))
            print("Are you sure you want to erase them and start fresh?")
            print("Warning: This will invalidate all training labels made from these clips!")
            if input("> ").lower().startswith('y'):
                for clip in existing_clips:
                    os.remove(os.path.join('clips', clip))
                from human_feedback_api import Clip
                Clip.objects.filter(environment_id=env_id).delete()
                # Also erase all label data for this experiment
                from human_feedback_api import SortTree
                SortTree.objects.filter(experiment_name=experiment_name).delete()
                from human_feedback_api import Comparison
                Comparison.objects.filter(experiment_name=experiment_name).delete()
            else:
                print("Quitting...")
                return

    if args.force_new_training_labels:
        from human_feedback_api import SortTree
        from human_feedback_api import Comparison
        all_tree_nodes = SortTree.objects.filter(experiment_name=experiment_name)
        if all_tree_nodes:
            print("Found a sorting tree with {} nodes".format(len(all_tree_nodes)))
            print("Are you sure you want to erase all the comparison data associated with this tree?")
            if input("> ").lower().startswith('y'):
                all_tree_nodes.delete()
                Comparison.objects.filter(experiment_name=experiment_name).delete()
            else:
                print("Quitting...")
                return

    print("Setting things up...")
    run_name = "%s/%s-%s" % (env_id, experiment_name, int(time()))
    env = make_env(env_id)
    n_pretrain_labels = args.pretrain_labels if args.pretrain_labels else (args.n_labels // 4 if args.n_labels else 0)

    episode_logger = EpisodeLogger(run_name)
    print("EPISODE LOGGER:: ", episode_logger)
    schedule = make_label_schedule(n_pretrain_labels, args.n_labels, args.num_timesteps, episode_logger)

    os.makedirs('checkpoints/reward_model', exist_ok=True)
    os.makedirs('clips', exist_ok=True)

    # Make reward model
    if args.reward_model == "rl":
        reward_model = OriginalEnvironmentReward(episode_logger)
        args.pretrain_iters = 0  # Don't bother pre-training a traditional RL agent
    else:
        reward_model = OrdinalRewardModel(
            args.reward_model, env, env_id, make_env, experiment_name, episode_logger, schedule,
            n_pretrain_labels, args.clip_length, args.stacked_frames, args.workers)

    if not args.force_new_reward_model:
        reward_model.try_to_load_model_from_checkpoint()

    reward_model.train(args.pretrain_iters, report_frequency=25)
    reward_model.save_model_checkpoint()

    # Wrap the reward model to capture videos every so often:
    if not args.no_videos:
        video_path = os.path.join('/tmp/rl_teacher_vids', run_name)
        # video_path = os.path.join('C:\\Users\\dgaspar\\Documents\\github\\test\\videos', run_name)
        checkpoint_interval = 20 if args.agent == "ga3c" else 200
        reward_model = SegmentVideoRecorder(reward_model, env, save_dir=video_path, checkpoint_interval=checkpoint_interval)

    print("Starting joint training of reward model and agent")
    if args.agent == "ga3c":
        from ga3c.Server import Server as Ga3cServer
        from ga3c.Config import Config as Ga3cConfig
        Ga3cConfig.ATARI_GAME = env_id
        Ga3cConfig.MAKE_ENV_FUNCTION = make_env
        Ga3cConfig.NETWORK_NAME = experiment_name
        Ga3cConfig.SAVE_FREQUENCY = 200
        Ga3cConfig.TENSORBOARD = True
        Ga3cConfig.LOG_WRITER = episode_logger
        Ga3cConfig.AGENTS = args.workers
        Ga3cConfig.LOAD_CHECKPOINT = not args.force_new_agent_model
        Ga3cConfig.STACKED_FRAMES = args.stacked_frames
        Ga3cConfig.BETA_START = args.starting_beta
        Ga3cConfig.BETA_END = args.starting_beta * 0.1
        Ga3cServer(reward_model).main()
    elif args.agent == "parallel_trpo":
        from parallel_trpo.train import train_parallel_trpo
        train_parallel_trpo(
            env_id=env_id,
            make_env=make_env,
            stacked_frames=args.stacked_frames,
            predictor=reward_model,
            summary_writer=episode_logger,
            workers=args.workers,
            runtime=(args.num_timesteps / 1000),
            max_timesteps_per_episode=get_timesteps_per_episode(env),
            timesteps_per_batch=8000,
            max_kl=0.001,
            seed=args.seed,
        )
    elif args.agent == "pposgd_mpi":
        from pposgd_mpi.run_mujoco import train_pposgd_mpi
        train_pposgd_mpi(lambda: make_env(env_id), num_timesteps=args.num_timesteps, seed=args.seed, predictor=reward_model)
    elif args.agent == "ppo_atari":
        from pposgd_mpi.run_atari import train_atari
        # TODO: Add Multi-CPU support!
        train_atari(env, num_timesteps=args.num_timesteps, seed=args.seed, num_cpu=1, predictor=reward_model)
    else:
        raise ValueError("%s is not a valid choice for args.agent" % args.agent)

if __name__ == '__main__':
    main()
