from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0.)

class RewardNet(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(RewardNet, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, sa):
        x = F.relu(self.linear1(sa))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

class DiscriminatorNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DiscriminatorNet, self).__init__()
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 2)

        self.apply(weights_init_)

    def forward(self, state, action, next_state):
        sas = torch.cat([state, action, next_state], 1)
        x = F.relu(self.linear1(sas))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        x = 2 * F.tanh(x) # TBD

        return x # regression label, unnormalized

class DiscNet(nn.Module):

    def __init__(self, skill_dim, state_dim, action_dim, disc_dims, hidden_dim, disc_mode=0):
        super(DiscNet, self).__init__()
        self.skill_dim = skill_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dims = disc_dims 
        self.disc_mode = disc_mode
        if disc_mode == 0 or disc_mode == 1:
            input_dim = state_dim + action_dim
        elif disc_mode == 2 or disc_mode == 3 :
            input_dim = action_dim
        elif disc_mode == 4 or disc_mode == 5 :
            input_dim = len(disc_dims)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, self.skill_dim)

        self.apply(weights_init_)

    def forward(self, S):
        if self.disc_mode == 0:
            ss = S[:, :, :].mean(1)
        elif self.disc_mode == 1:
            ss = torch.cat((S[:, 0, :self.action_dim], S[:, 1, self.action_dim:]), -1)
        elif self.disc_mode == 2:
            ss = S[:, :, :self.action_dim].mean(1)
        elif self.disc_mode == 3:
            ss = S[:, 0, :self.action_dim]#.mean(1)
        elif self.disc_mode == 4:
            ss = S[:, :, self.action_dim:][:,:,self.dims].mean(1)
        elif self.disc_mode == 5:
            ss = S[:, :, self.action_dim:][:,1,self.dims]#.mean(1)
        x = self.linear1(ss)
        # x = F.relu(self.linear2(x))
        x = self.linear4(x)
        return x


class ActionNet(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ActionNet, self).__init__()

        self.linear1 = nn.Linear(state_dim+state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)

    def forward(self, ss):
        x = F.relu(self.linear1(ss))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x