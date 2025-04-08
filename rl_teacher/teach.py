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

from ga3c.Server import Server as Ga3cServer
from ga3c.Config import Config as Ga3cConfig

from threading import Thread

import gym
from ui_adapt.envs.uiadaptationenv import UIAdaptationEnv
from ui_adapt.utils import StopImage
# from ui_adapt.RL_algorithms import QLearningAgent as RLagent

import asyncio

def main(env=None, user_id=None, args=None):
    # Tensorflow is not fork-safe, so we must use spawn instead
    # https://github.com/tensorflow/tensorflow/issues/5448#issuecomment-258934405
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass  # context has already been set

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
    env = make_env(env_id, env=env)
    n_pretrain_labels = args.pretrain_labels if args.pretrain_labels else (args.n_labels // 4 if args.n_labels else 0)

    episode_logger = EpisodeLogger(run_name)
    print("EPISODE LOGGER:: ", episode_logger)
    schedule = make_label_schedule(n_pretrain_labels, args.n_labels, args.num_timesteps, episode_logger)

    os.makedirs('checkpoints/reward_model', exist_ok=True)
    os.makedirs('checkpoints/agent', exist_ok=True)
    os.makedirs('clips', exist_ok=True)

    # Make reward model
    if args.reward_model == "rl":
        reward_model = OriginalEnvironmentReward(episode_logger)
        args.pretrain_iters = 0  # Don't bother pre-training a traditional RL agent
    else:
        print("STARTING THE ORDINAL RM")
        domain = args.domain
        user_id_arg = args.user_id
        print("Domain::: ", domain)
        print("Play mode: ", args.play_mode)
        training = not args.play_mode
        if args.general_model:
            user_id_arg = 8
            if args.general_model_uid:
                user_id_arg = args.general_model_uid
        reward_model = OrdinalRewardModel(
            args.reward_model, env, env_id, make_env, experiment_name, episode_logger, schedule,
            n_pretrain_labels, args.clip_length, args.stacked_frames, args.workers,
            user_id=user_id_arg, domain=domain, training=training)
    if not args.force_new_reward_model:
        print("TRYING TO LOAD THE REWARD MODEL.")
        reward_model.try_to_load_model_from_checkpoint()
        print(reward_model)
        print("REWARD MODEL FINISHED.")
    if training: 
        print("REWARD MODEL FINISHED. TRAINNING STARTING.")
        reward_model.train(args.pretrain_iters, report_frequency=25)
        reward_model.save_model_checkpoint()
        reward_model.clip_manager.free_memory()

    # Wrap the reward model to capture videos every so often:
    if not args.no_videos:
        # video_path = os.path.join('C:', 'tmp', 'rl_teacher_media', run_name)
        video_path = os.path.join('/tmp/rl_teacher_vids', run_name)
        # video_path = os.path.join('C:\\Users\\dgaspar\\Documents\\github\\test\\videos', run_name)
        checkpoint_interval = 20 if args.agent == "ga3c" else 200
        reward_model = SegmentVideoRecorder(reward_model, env, save_dir=video_path, checkpoint_interval=checkpoint_interval)


    print("Starting joint training of reward model and agent")
    if args.agent == "ga3c":        
        Ga3cConfig.ATARI_GAME = env_id
        Ga3cConfig.MAKE_ENV_FUNCTION = make_env
        Ga3cConfig.NETWORK_NAME = experiment_name
        Ga3cConfig.SAVE_FREQUENCY = 5
        Ga3cConfig.TENSORBOARD = True
        Ga3cConfig.LOG_WRITER = episode_logger
        Ga3cConfig.AGENTS = args.workers
        Ga3cConfig.TENSORBOARD_UPDATE_FREQUENCY = 20
        Ga3cConfig.LOAD_CHECKPOINT = not args.force_new_agent_model
        Ga3cConfig.STACKED_FRAMES = args.stacked_frames
        Ga3cConfig.BETA_START = args.starting_beta
        Ga3cConfig.BETA_END = args.starting_beta * 0.1
        Ga3cConfig.ENV = env
        Ga3cConfig.EPISODES = args.episodes

        Ga3cConfig.PLAY_MODE = False
        Ga3cConfig.TRAIN_MODELS = True
        # Ga3cConfig.PLAY_MODE = True
        # Ga3cConfig.TRAIN_MODELS = False
        if args.play_mode:
            print("\nGOING WITH THE PLAY MODE!!!\n")
            Ga3cConfig.AGENTS = 1
            Ga3cConfig.PREDICTORS = 1
            Ga3cConfig.TRAINERS = 1
            Ga3cConfig.DYNAMIC_SETTINGS = False

            Ga3cConfig.LOAD_CHECKPOINT = True
            Ga3cConfig.TRAIN_MODELS = False
            Ga3cConfig.SAVE_MODELS = False
            
            Ga3cConfig.PLAY_MODE = True
        if user_id:
            print("running it for UID: ", args.user_id)
            agents[user_id] = Ga3cServer(reward_model)
            agents[user_id].main()
        else:
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
    elif args.agent == "MCTS":
        pass
    else:
        raise ValueError("%s is not a valid choice for args.agent" % args.agent)
    print("THE TEACH 'MAIN' HAS FINISHED.")


import json
# Called for every client connecting (after handshake)
def new_client(client, server):
    print("New client connected and was given id %d" % client['id'])
    print(f"now this is our client list: {server.clients}")
    message_init = {
        "type": "init",
        "value": ""
        }
    message_init_string = json.dumps(message_init)
    server.send_message(client, message_init_string)


## Global vars
agents = {}
envs = {}
configs = {}
users = {}


# Called for every client disconnecting
def client_left(client, server):
    print("Client(%d) disconnected" % client['id'])

# Called when a client sends a message
def message_received(client, server, message):
    json_message = json.loads(message)
    if "type" in json_message and json_message["type"] == "init_response":
        prev_user_exists = False
        user_id = json_message.get("user_id", None)
        domain = json_message.get("domain", None)
        db_user_id = json_message.get("db_user_id", None)
        mode = json_message.get("play_mode", False)

        return_message = {
            "type": "init_ready",
            "value": "Agent created with " + str(
                json_message["algorithm"]) + " algorithm",
            "algorithm": json_message["algorithm"],
            "reward_type": json_message["state"]["REWARD"]["MODE"],
            "updateStatus": "False"
        }
        
        if user_id in users:
            prev_user_exists = True
            envs[users[user_id]].updateSockets(client, server)
            return_message["updateStatus"] = "True"
        else:
            initial_state = json_message["state"]
            config = createConfigJSON(json_message, client)
            configs[client["id"]] = config
            env = createEnv(client, config, initial_state, client, server, domain, user_id)
            print(f"Agent created for Client {client['id']}.\nAgent list: {agents}.\nEnvironments List: {envs}")
            users[json_message["user_id"]] = client['id']
        
        return_message_string = json.dumps(return_message)
        server.send_message(client, return_message_string)

        if not prev_user_exists:
            print(f"Environment created for Client {client['id']}.")
            print("Starting the learning process!")
            
            if domain:
                args.domain = domain
            if db_user_id != -1:
                args.user_id = db_user_id

            args.play_mode = mode
            thread = Thread(target = main, args = (env, users[json_message["user_id"]], args))
            thread.start()

            print("RUNNING MAIN: ", thread)

    elif "type" in json_message and json_message["type"] == "returnImage":
        uid = users[json_message["user_id"]]
        if isinstance(json_message["value"], str) and json_message["value"] == "stop": 
            envs[uid].setImage(StopImage())
        else:
            envs[uid].setImage(json_message["value"])
    elif "type" in json_message and json_message["type"] == "askAdaptation":
        initial_state = json_message["state"]
        config = createConfigJSON(json_message, client)
        uid = users[json_message["user_id"]]

        # Check if the agent is a Ga3cServer instance
        print("agents: ", agents)
        if isinstance(agents[uid], Ga3cServer):
            ask_image_msg = {
                "type": "image",
            }
            ask_image_msg_string = json.dumps(ask_image_msg)
            server.send_message(client, ask_image_msg_string)
        else:
            agents[uid].adapt(config=config, initial_state=initial_state, env=envs[uid])
    elif "type" in json_message and json_message["type"] == "takeAction":
        action_id_number = -1
        uid = users[json_message["user_id"]]

        for action_id, action_data in configs[uid]["ACTIONS"].items():
            if action_id == "MODE":
                continue
            if action_data["value"] == json_message["value"]:
                # Print the corresponding action ID
                print("Action ID for", json_message["value"] + ":", action_id)
                action_id_number = action_id
                break
        else:
            print("Action", json_message["value"], "not found in config.")
        uid = users[json_message["user_id"]]
        print("THIS IS THE AGENT::: ", agents[uid])
        envs[uid].step(int(action_id_number))
        print("THIS IS THE AGENT::: ", agents[uid])

    else:
        print("Client(%d) said: %s" % (client['id'], message))


def createEnv(client, config, initial_state, client_socket, server_socket, domain, user_id):
    env = UIAdaptationEnv(config_data=config, 
                          initialState=initial_state,
                          ws_client=client_socket,
                          ws_server=server_socket,
                          learning=True)
    envs[client["id"]] = env
    return env

def createConfigJSON(config_info, client):
    config = {}
    createContext(config)
    createUIDesign(config, config_info["value"])
    createActions(config)
    if "REWARD" in config_info["state"]:
        createReward(config, config_info["state"]["REWARD"])
    createAPIConnection(config, client)
    return config

def createUIDesign(config, ui_description):
    config["UIDESIGN"] = {}
    for factor in ui_description:
        config["UIDESIGN"][factor.upper()] = ui_description[factor]
    return config

def createReward(config, config_info):
    config["REWARD"] = {
        "MODE": config_info["MODE"]
    }

def createActions(config):
    config["ACTIONS"] = {}
    config["ACTIONS"]["MODE"] = "WEBSOCKET"
    action = 0
    config["ACTIONS"][str(action)] = {
        "name": "No operate",
        "target": "pass",
        "value": "pass",
        "api_call": "adaptation pass pass"
        }
    action += 1
    all_adaptations = config["UIDESIGN"]
    for factor in all_adaptations:
        for val in all_adaptations[factor]:
            config["ACTIONS"][str(action)] = {
                "name": "Change to " + str(val).lower(),
                "target": str(factor).lower(),
                "value": str(val).lower(),
                "api_call": "adaptation " + str(factor).lower()  + " " + str(val).lower()
            }
            action += 1

    return config

def createAPIConnection(config, client):
    config["API_CONNECTION"] = {
        "HOST": client['address'][0],
        "PORT": client['address'][1],
        "RESOURCES": "",
        "RENDER_RESOURCE": ""
    }

def createContext(config):
    config["USER"] = {
        "AGE": ["noObt"]
    }
    config["PLATFORM"] = {
        "DEVICE": ["noObt"]
    }
    config["ENVIRONMENT"] = {
        "LOCATION": ["noObt"]
    }
    return config

def run_server(args=None):
    from websocket_server import WebsocketServer
    if args.port:
        PORT = args.port
    else:
        PORT=9997
    HOST="127.0.0.1"
    server = WebsocketServer(host = HOST, port = PORT)
    server.set_fn_new_client(new_client)
    server.set_fn_client_left(client_left)
    server.set_fn_message_received(message_received)
    print(F"Server running on {HOST}:{PORT}")
    server.run_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', required=False, type=str)
    parser.add_argument('-play_mode', '--play_mode', default=False)
    parser.add_argument('-e', '--env_id', required=True)
    parser.add_argument('-d', '--domain', required=False, type=str)
    parser.add_argument('-u', '--user_id', required=False, type=int)
    parser.add_argument('-p', '--reward_model', required=True)
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-s', '--seed', default=1, type=int)
    parser.add_argument('-w', '--workers', default=4, type=int)
    parser.add_argument('-l', '--n_labels', default=None, type=int)
    parser.add_argument('-L', '--pretrain_labels', default=None, type=int)
    parser.add_argument('-t', '--num_timesteps', default=5e6, type=int)
    parser.add_argument('-a', '--agent', default="ga3c", type=str)
    parser.add_argument('-i', '--pretrain_iters', default=5000, type=int)
    parser.add_argument('-tep', '--episodes', default=400000, type=int)
    parser.add_argument('-b', '--starting_beta', default=0.1, type=float)
    parser.add_argument('-c', '--clip_length', default=1.5, type=float)
    parser.add_argument('-f', '--stacked_frames', default=4, type=int)
    parser.add_argument('-V', '--no_videos', action="store_true")
    parser.add_argument('--force_new_environment_clips', action="store_true")
    parser.add_argument('--force_new_training_labels', action="store_true")
    parser.add_argument('--force_new_reward_model', action="store_true")
    parser.add_argument('--force_new_agent_model', action="store_true")
    parser.add_argument('--port', '--port', required=False, type=int)
    parser.add_argument('--general_model', action="store_true")
    parser.add_argument('-guid', '--general_model_uid', default=8, type=int)
    args = parser.parse_args()

    if args.mode and args.mode == 'server':
        run_server(args=args)
    else:
        main(args=args)

