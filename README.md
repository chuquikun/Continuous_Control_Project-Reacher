# P2: Continuous Control Project - Reacher
<p align="center">
<img align="center" src="https://github.com/chuquikun/Continuous_Control_Project-Reacher/blob/main/images/trained_reachers.gif">
</p>
<p align="center">DDPG Tranied Reacher Agents</p>

### Introduction

This project uses the Reacher environment. In this, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

This repository works with the environment version that runs 20 parallel agents. In particular, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
This yields an average score for each episode (where the average is over all 20 agents).
As an example, consider the plot below, where we have plotted the average score (over all 20 agents) obtained with each episode.

### Setting the environment up

1. To run the environment we need a specific version of ptyhon so you are ancouraged to set a conda environment. Here the env was called dqn_bca (deep q network banana collector agent):
* For Linux or Mac:

```bash
conda create --name drlnd python=3.6
conda activate drlnd
```
* For Windows:
```bash
conda create --name drlnd python=3.6 
activate drlnd
```

2. Perform a minimal installation of OpenAI gym:

*  Run the following line to perform minimal installation
```
git clone https://github.com/openai/gym.git
cd gym
pip install -e .
```
* Install the **classic control** and **box2** environments by runnning:
```
pip install -e '.[classic_control]'
pip install -e '.[box2d]'
```

3. Clone or download this repository:
```
https://github.com/chuquikun/Continuous_Control_Project-Reacher.git
```
* Move to the folder `python/ ` and install dependencies within:
```
cd python
pip install .
```
4. Create an IPython kernel for the drlnd environment:

```
python -m ipykernel install --user --name drlnd --display-name "drlnd".
```
5. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.
    
    Alternatively you can download a headless version of the Linux environment which is very convenient to train the agent without launch the graphic interface.
    - Linux No Visualization:[click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip)
    
6. Place the file in the the roor of this repository and unzip (or decompress) the file. 

7. Running the notebooks 

To launch the notebooks run in the root of this directory:
```
jupyter notebook
```
Finally select and double-click the notebook you want to run.
Before running code in any notebook, change the kernel to match the drlnd environment by using the drop-down Kernel menu.

### Instructions

This repository contains 4 main files:

- **model.py** contains the code to train the neural networks that represent the Actor and Critic for the DDPG agent.
- **ddpg_agent.py** decribes the Agent class and contains how the agents act and learn.
- **continuous_control_ddpg.ipynb** this is the notebook you may want to run to train a new agent or modify the existing one.
- **run_trained_agent.ipynb** this is the notebook that you need to run if you want to see the performance of the already trained agents.
- **utils.py" contains a utility function to load  a trained agent.

The file Report.md contains an a briefly explanation of the algorithms used to train the agent.

