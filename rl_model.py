import numpy as np
import torch
from torch.autograd import Variable
from torch import autograd, optim, nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple


class AC_MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super(AC_MLP, self).__init__()
        assert type(hidden_size) == list and len(hidden_size) == 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size[0])
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.sigmoid = torch.nn.Sigmoid()
        self.actor = torch.nn.Linear(self.hidden_size[1], self.action_size)
        self.critic = torch.nn.Linear(self.hidden_size[1], 1)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = x.to(self.device)
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        policy = F.softmax(self.actor(output), dim=1)
        value = self.critic(output)
        return policy, value


class AC_RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_LSTM_layers, action_size):
        super(AC_RNN, self).__init__()
        assert type(hidden_size) == list and len(hidden_size) == 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.num_LSTM_layers = num_LSTM_layers  # number of LSTM layers
        self.lstm = torch.nn.LSTM(self.input_size, self.hidden_size[0], self.num_LSTM_layers)  # TODO: add a dropout layer?
        self.linear = torch.nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.relu = torch.nn.ReLU()
        self.actor = torch.nn.Linear(self.hidden_size[1], self.action_size)
        self.critic = torch.nn.Linear(self.hidden_size[1], 1)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.saved_actions = []
        self.rewards = []

    def reinit_hid(self):
        self.hidden = (torch.randn(self.num_LSTM_layers, self.batch_size, self.hidden_size[0]).to(self.device),  # hx: (#layers, hidden_size)
                       torch.randn(self.num_LSTM_layers, self.batch_size, self.hidden_size[0]).to(self.device))  # cx: (#layers, hidden_size)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[-1] == self.input_size
        assert x.shape[-2] == self.batch_size
        x = x.to(self.device)
        out, self.hidden = self.lstm(x, self.hidden)
        output = self.linear(out)
        output = self.relu(output)
        policy = F.softmax(self.actor(output), dim=1)
        value = self.critic(output)
        return policy, value


# ======================================

# ---------- helper functions ----------

# ======================================

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


def select_action(model, policy_, value_):
    a = Categorical(policy_)
    action = a.sample()
    model.saved_actions.append(SavedAction(a.log_prob(action), value_))
    return action.item(), policy_.data[0], value_.item()


def discount_rwds(r, gamma):  # takes [1,1,1,1] and makes it [3.439,2.71,1.9,1]
    disc_rwds = np.zeros_like(r).astype(float)
    r_asfloat = r.astype(float)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r_asfloat[t]
        disc_rwds[t] = running_add
    return disc_rwds


def finish_trial(model, discount_factor, optimizer, **kwargs):
    '''
    Finishes a given training trial and backpropagates.
    '''

    # set the return to zero
    R = 0
    returns_ = discount_rwds(np.asarray(model.rewards), gamma=discount_factor)  # [1,1,1,1] into [3.439,2.71,1.9,1]
    saved_actions = model.saved_actions

    policy_losses = []
    value_losses = []

    returns_ = torch.Tensor(returns_).to(model.device)

    for (log_prob, value), r in zip(saved_actions, returns_):
        rpe = r - value.item()
        policy_losses.append(-log_prob * rpe)
        value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([[r]]).to(model.device))).unsqueeze(-1))
        #   return policy_losses, value_losses
    optimizer.zero_grad() # clear gradient
    p_loss = (torch.cat(policy_losses).sum())
    v_loss = (torch.cat(value_losses).sum())
    total_loss = p_loss + v_loss
    total_loss.backward(retain_graph=True) # calculate gradient
    optimizer.step()  # move down gradient

    del model.rewards[:]
    del model.saved_actions[:]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return p_loss, v_loss

