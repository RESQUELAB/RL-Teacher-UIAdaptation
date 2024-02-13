# RL-Teacher-UI-Adapt

`rl-teacher-ui-adapt` is an extension of [`rl-teacher-ateri`](https://github.com/machine-intelligence/rl-teacher-atari) whis is in turn an extension of [`rl-teacher`](https://github.com/nottombrown/rl-teacher), which is in turn an implementation of of [*Deep Reinforcement Learning from Human Preferences*](https://arxiv.org/abs/1706.03741) [Christiano et al., 2017].

As-is, `rl-teacher` only handles MuJoCo environments. This repository is meant to extend that functionality to User Interface adaptation environments. For example ['UI-adaptation-RL-env'](https://github.com/ISSI-DSIC/UI-adaptation-RL-env). 

This repository aims to handle `Mujoco`, `Atari` and `UI-Adaptation` environments.

# Installation

The setup instructions are identical to [`rl-teacher`](https://github.com/nottombrown/rl-teacher#installation).

## 1. Set up an anaconda environment

- ```conda create -n UIrlhf python==3.6.3```
- ```conda activate UIrlhf```

## 2. Install dependencies

```
cd ~/rl-teacher-ui-adapt
pip install -e .
pip install -e human-feedback-api
pip install -e agents/ga3c
```

## 3. Install existing Environments

### 3.1 Atari Enviroments

Dependencies: ```pip install atari-py```

You might need to Import the ROMS. To import the ROMS, you need to download `Roms.rar` from the [Atari 2600 VCS ROM Collection website](http://www.atarimania.com/roms/) and extract the .rar file. You can download the file using the following command:
`wget http://www.atarimania.com/roms/Roms.rar`

After downloading and extracting the Roms file, you must run

```python -m atari_py.import_roms <path to folder>```

 to import them. This command should print out the names of ROMs as it imports them. The ROMs will be copied to your atari_py installation directory.


### 3.2 MuJoCo Environments

#### 3.2.1 Install Mujoco and `mujoco-py`

1. Download the MuJoCo version 2.1 binaries 
   [here](https://www.roboti.us/download/mjpro131_linux.zip).
2. Extract the downloaded `mjpro131` directory into `~/.mujoco/mjpro131`.

3. Install dependencies `pip install "cython<3"`

4. Make sure that you have the following libraries installed:
`sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3`

#### 3.2.2 Try Mujoco 


```
cd ~/.mujoco/mjpro131/bin
./simulate ../model/humanoid100.xml
```

#### 3.2.3 Add Mujoco Licence


`wget https://www.roboti.us/file/mjkey.txt -i ~/.mujoco/mjkey.txt`


### 3.3 User Interface Adaptation Environments

ToDo

## 4. Run Existing Environments 

### 4.1 Atari


- Baseline RL: ```python rl_teacher/teach.py -e Pong-v0 -n rl-test -p rl```
- Synthetic data RL: ```python rl_teacher/teach.py -e Breakout-v0 -n synth-test -p synth -l 300```
- Huam Feedback: ```python rl_teacher/teach.py -e Breakout-v0 -n human-test -p human -L 50```


### 4.2 MuJoCo

ToDo


### 4.3 UI-Adapt

ToDo

### 4.4 Summary

- For Baseline RL
  - ```python rl_teacher/teach.py -e <ENV-name> -n <Experiment-Name> -p **rl**```
- For Synthetic labels RL
  - ```python rl_teacher/teach.py -e <ENV-name> -n <Experiment-Name> -p **synth** -l <Number of labels>```
- For Human Feedback RL
  - ```python rl_teacher/teach.py -e <ENV-name> -n <Experiment-Name> -p **human** -L <Number of pre-train labels>```



# 5. Monitoring the Agents with Tensorboard

By default, this will write tensorboard files to `~/tb/rl-teacher/base-rl`. Start tensorboard as follows:

    $ tensorboard --logdir ~/tb/rl-teacher/
    Starting TensorBoard b'47' at http://0.0.0.0:6006
    (Press CTRL+C to quit)

Navigate to http://0.0.0.0:6006 in a browser to view your learning curves.


# 6. Human labels

To train your agent based off of feedback from a real human, you’ll run two separate processes:

1. The agent training process. This is very similar to the commands we ran above.
2. A webapp, which will show you short video clips of trajectories and ask you to rate which clip is better.

## 6.1 Set up the `human-feedback-api` webapp
First you'll need to set up django. This will create a `db.sqlite3` in your local directory.
```
python human-feedback-api/manage.py migrate
python human-feedback-api/manage.py collectstatic
```

Start the webapp

```python human-feedback-api/manage.py runserver 0.0.0.0:8000```

You should now be able to open the webapp by navigating to http://localhost:8000/ in any browser. There’s nothing there yet, but when you run your agent, it will create an experiment that will let you add labels.

## 6.2 Create a Place to store rendered trajectory segments

### Option 1: Create a Flask Server to store rendered trajectory segments

Start the localhost server using

```python human-feedback-api/video_server/run_server.py```

Open a Web broser and go to [127.0.0.1:5000/](127.0.0.1:5000/). You should see a confirmation message that the Server is working.


### Option 2: Create a GCS bucket to store rendered trajectory segments

#### (Not compatible with local storage - Current version does not support GCS)

The training process generates rendered trajectory segments for you to provide feedback on. These are stored in Google Cloud Storage (GCS), so you will need to set up a GCS bucket.

If you don't already have GCS set up, [create a new GCS account](https://cloud.google.com/storage/docs/) and set up a new project. Then, use the following commands to create a bucket to host your media and set this new bucket to be publicly-readable.

    export RL_TEACHER_GCS_BUCKET="gs://rl-teacher-<YOUR_NAME>"
    gsutil mb $RL_TEACHER_GCS_BUCKET
    gsutil defacl ch -u AllUsers:R $RL_TEACHER_GCS_BUCKET

## 6.3 Run your agent
Now we're ready to train an agent with human feedback!

Note: if you have access to a remote server, we highly recommend running the agent training remotely, and provide feedback in the webapp locally. You can run both the agent training and the feedback app on your local machine at the same time. However, it will be annoying, because the rendering process during training will often steal window focus. For more information on running the agent training remotely, see the [Remote Server instructions](#using-a-remote-server-for-agent-training) below.

Run the command below to start the agent training. The agent will start to take random actions in the environment, and will generate example trajectory segments for you to label:

```python rl_teacher/teach.py -p human --pretrain_labels 175 -e Pong-v0 -n human-175```

Once the training process has generated videos for the trajectories it wants you to label, you will see it uploading these to GCS:

    ...
    Copying media to gs://rl-teacher-catherio/d659f8b4-c701-4eab-8358-9bd532a1661b-right.mp4 in a background process
    Copying media to gs://rl-teacher-catherio/9ce75215-66e7-439d-98c9-39e636ebb8a4-left.mp4 in a background process
    ...

In the meantime the agent training will pause, and wait for your feedback:

    0/175 comparisons labeled. Please add labels w/ the human-feedback-api. Sleeping...


## 6.4 Provide feedback to agent

At this point, you can click on the active experiment to enter the labeling interface. Click the Active Experiment link.


Once you are in the labeling interface, you will see pairs of clips. For each pair, indicate which one shows better behavior, for whatever you are trying to teach the agent to do. (To start with, you might try to teach Reacher how to spin counter-clockwise, or come up with your own task!)


Once you have finished labeling the 175 pretraining comparisons, we train the predictor to convergence on the initial comparisons. After that, it will request additional comparisons every few seconds.

If you see a blank screen like this at any point, it means the clip is not yet ready to display. Try waiting a few minutes and refreshing the page, or click `Can't tell` to move on and try another clip

That's it! The more feedback you provide, the better your agent will get at the task.



# 7. Notes:

- You should use the `parallel_trpo` agent for solving MuJoCo environments
- You should use the `ga3c` agent for solving Atari environments

For example: 
```
python rl_teacher/teach.py -p rl -e ShortHopper-v1 -n base-rl-mujoco -a parallel_trpo
python rl_teacher/teach.py -p rl -e Pong-v0 -n base-rl-atari -a ga3c
```

There are a few new command-line arguments that are worth knowing about. Primarily, there are a set of four flags:
- `--force_new_environment_clips`
- `--force_new_training_labels`
- `--force_new_reward_model`
- `--force_new_agent_model`
Activating these flags will erase the corresponding data from the disk/database. For the most part this won't be necessary, and you can simply pick a new experiment name. Note, however, that *experiments within the same environment now share clips* so you may want to `--force_new_environment_clips` when starting a new experiment in an old environment.

Also worth noting, there's a parameter called `--stacked_frames` (`-f`) that defaults to *4*. This helps model movement that the human naturally sees in the video, but can alter how the system performs compared to `rl-teacher`. To remove frame stacking simply add `-f 0` to the command-line arguments.

## Backwards Compatibility

`rl-teacher-ui-adapt` is meant to be entirely backwards compatible, and do at least as well as `rl-teacher-atari` and `rl-teacher` on all tasks. If `rl-teacher-ui-adapt` lacks a feature that its parent has, please submit an issue.



