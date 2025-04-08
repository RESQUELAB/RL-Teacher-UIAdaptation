"""
This is an experimental file used to take multiple trained models and see how they perform in an environment.
Unlike most of the codebase, this is provided as-is, without any guarantee of working.
"""

import numpy as np

import gym
from gym import wrappers

from ga3c.NetworkVP import NetworkVP
from ga3c.Environment import Environment
from ga3c.Config import Config as Ga3cConfig
from rl_teacher.reward_models import OriginalEnvironmentReward, OrdinalRewardModel


np.set_printoptions(precision=2, linewidth=150)

if __name__ == '__main__':
    env_id = 'UIAdaptation-v0'
    domain = "courses"
    user_id = "63"
    reward_model = OrdinalRewardModel(
        args.reward_model, env, env_id, make_env, experiment_name, episode_logger, schedule,
        n_pretrain_labels, args.clip_length, args.stacked_frames, args.workers, user_id=user_id_arg, domain=domain)
    if not args.force_new_reward_model:
        reward_model.try_to_load_model_from_checkpoint()
    print("REWARD MODEL FINISHED. TRAINNING STARTING.")
    reward_model.train(args.pretrain_iters, report_frequency=25)
    reward_model.save_model_checkpoint()

    # Wrap the reward model to capture videos every so often:
    if not args.no_videos:
        # video_path = os.path.join('C:', 'tmp', 'rl_teacher_media', run_name)
        video_path = os.path.join('/tmp/rl_teacher_vids', run_name)
        # video_path = os.path.join('C:\\Users\\dgaspar\\Documents\\github\\test\\videos', run_name)
        checkpoint_interval = 20 if args.agent == "ga3c" else 200
        reward_model = SegmentVideoRecorder(reward_model, env, save_dir=video_path, checkpoint_interval=checkpoint_interval)




    model_names = ['left']

    Ga3cConfig.ATARI_GAME = env_id
    Ga3cConfig.MAKE_ENV_FUNCTION = gym.make
    Ga3cConfig.PLAY_MODE = True

    env = Environment()
    actions = np.arange(env.get_num_actions())
    done = False
    command = None
    command_steps = -1

    preprogrammed_sequences = {
        'BOTTOM_RIGHT': [11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]}

    models = {name: NetworkVP('cpu:0', name, len(actions)) for name in model_names}
    for model in models.values():
        model.load()

    while not done:
        if env.current_state is None:
            env.step(0)  # NO-OP while we wait for the frame buffer to fill.
        else:
            if command_steps > 0:
                command_steps -= 1
                if command.isdigit():
                    action = int(command)
                else:
                    model = models[command]
                    p = model.predict_p(np.expand_dims(env.current_state, axis=0))[0]
                    # action = np.argmax(p)
                    action = np.random.choice(actions, p=p)
                _, done, _ = env.step(action)
            else:
                if command is None:
                    print('Please input commands with a number of steps (like "left 12")')
                raw = input().split()
                if len(raw) == 1 and raw[0] in preprogrammed_sequences:
                    for action in preprogrammed_sequences[raw[0]]:
                        env.step(action)
                elif len(raw) != 2:
                    print('Malformed input. Try again.')
                else:
                    command, command_steps = raw[0], int(raw[1])
                    if command not in models and not command.isdigit():
                        print('Unknown command "%s"' % command)
                        command_steps = -1
