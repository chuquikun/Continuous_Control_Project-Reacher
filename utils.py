import torch
from ddpg_agent import Agent

def load_trained_agent(filepath):
    """ Load the results an parameters of a trained agent"""
    checkpoint = torch.load(filepath)
    agent = Agent(state_size=checkpoint['state_size'],
                 action_size=checkpoint['action_size'],
                 random_seed=checkpoint['seed'],
                 hidden_layers=checkpoint['hidden_layers'],
                 n_agents=checkpoint['n_agents'])
    
    agent.actor_local.load_state_dict(checkpoint['al_state_dict'])
    agent.critic_local.load_state_dict(checkpoint['cl_state_dict'])
    
    return agent

