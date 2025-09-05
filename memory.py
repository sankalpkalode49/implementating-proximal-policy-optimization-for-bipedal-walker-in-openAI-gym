import torch

class Memory:
    """
    A class to store the experiences of the agent for a single batch.
    """
    def __init__(self):
        # Initialize empty lists to store all the necessary information
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def store(self, state, action, reward, done, log_prob, value):
        """
        Stores a single step of experience.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        """
        Clears all the stored experiences after an update.
        """
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.dones[:]
        del self.log_probs[:]
        del self.values[:]
        
    def get_tensors(self):
        """
        Converts the stored lists of experiences into PyTorch tensors.
        This is useful for feeding the data into the neural network.
        """
        states = torch.tensor(self.states, dtype=torch.float32)
        actions = torch.tensor(self.actions, dtype=torch.float32)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)
        log_probs = torch.tensor(self.log_probs, dtype=torch.float32)
        values = torch.tensor(self.values, dtype=torch.float32)
        
        return states, actions, rewards, dones, log_probs, values