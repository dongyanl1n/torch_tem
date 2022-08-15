import numpy as np
import torch
from torch.autograd import Variable
from torch import autograd, optim, nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple


class actor_critic_agent(nn.Module):
    """
    An actor-critic neural network that takes 1d sensory input and generates a policy and a value function.
    """

    def __init__(self, input_dimensions, action_dimensions, batch_size, hidden_types, hidden_dimensions):
        """
           Create an actor-critic network class.
           - input_dimensions (int): the dimensions of the input space
           - action_dimensions (int): the number of possible actions
           - batch_size (int): the size of the batches.
           - hidden_types (list of strings): the type of hidden layers to use, options are 'linear', 'lstm', 'gru'.
           If list is empty no hidden layers are used.
           - hidden_dimensions (list of ints): the dimensions of the hidden layers. Must be a list of
                                           equal length to hidden_types.
       """

        super(actor_critic_agent, self).__init__()
        self.input_d = input_dimensions
        assert len(hidden_types) > 0, "must have hidden layer"
        self.hidden_types = hidden_types
        self.batch_size = batch_size
        assert len(hidden_types) is len(hidden_dimensions)

        # use gpu
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

        # to store a record of the last hidden states
        self.hx = []
        self.cx = []
        # create the hidden layers
        self.hidden = nn.ModuleList()
        for i, htype in enumerate(hidden_types):
            # check if hidden layer type is correct
            assert htype in ['linear', 'lstm', 'gru'], "hidden layer must be linear, lstm or gru"
            # get the input dimensions
            # first hidden layer
            if i is 0:
                input_d = input_dimensions
                output_d = hidden_dimensions[i]
                if htype is 'linear':
                    self.hidden.append(nn.Linear(input_d, output_d))
                    self.hx.append(None)
                    self.cx.append(None)
                elif htype is 'lstm':
                    self.hidden.append(nn.LSTMCell(input_d, output_d))
                    self.hx.append(Variable(torch.zeros(self.batch_size, output_d)).to(self.device))
                    self.cx.append(Variable(torch.zeros(self.batch_size, output_d)).to(self.device))
                elif htype is 'gru':
                    self.hidden.append(nn.GRUCell(input_d, output_d))
                    self.hx.append(Variable(torch.zeros(self.batch_size, output_d)).to(self.device))
                    self.cx.append(None)
            # second hidden layer onwards
            else:
                input_d = hidden_dimensions[i - 1]
                # get the output dimension
                output_d = hidden_dimensions[i]
                # construct the layer
                if htype is 'linear':
                    self.hidden.append(nn.Linear(input_d, output_d))
                    self.hx.append(None)
                    self.cx.append(None)
                elif htype is 'lstm':
                    self.hidden.append(nn.LSTMCell(input_d, output_d))
                    self.hx.append(Variable(torch.zeros(self.batch_size, output_d)))
                    self.cx.append(Variable(torch.zeros(self.batch_size, output_d)))
                elif htype is 'gru':
                    self.hidden.append(nn.GRUCell(input_d, output_d))
                    self.hx.append(Variable(torch.zeros(self.batch_size, output_d)))
                    self.cx.append(None)
        # create the actor and critic layers
        self.layers = [input_dimensions] + hidden_dimensions + [action_dimensions]
        self.output = nn.ModuleList([
            nn.Linear(output_d, action_dimensions),  # actor
            nn.Linear(output_d, 1)  # critic
        ])
        # store the output dimensions
        self.output_d = output_d
        # to store a record of actions and rewards
        self.saved_actions = []
        self.rewards = []

        self.to(self.device)

    def forward(self, x):
        '''
        forward(x):
        Runs a forward pass through the network to get a policy and value.
        Required arguments:
          - x (torch.Tensor): sensory input to the network, should be of size batch x input_d
        '''

        # check the inputs
        assert x.shape[-1] == self.input_d
        x = x.to(self.device)
        # pass the data through each hidden layer
        for i, layer in enumerate(self.hidden):
            # run input through the layer depending on type
            if isinstance(layer, nn.Linear):
                x = F.relu(layer(x))
            elif isinstance(layer, nn.LSTMCell):
                x, cx = layer(x, (self.hx[i], self.cx[i]))
                self.hx[i] = x.clone()
                self.cx[i] = cx.clone()
            elif isinstance(layer, nn.GRUCell):
                x = layer(x, self.hx[i])
                self.hx[i] = x.clone()
        # pass to the output layers
        policy = F.softmax(self.output[0](x), dim=1)
        value = self.output[1](x)

        return policy, value

    def reinit_hid(self):
        # to store a record of the last hidden states
        self.hx = []
        self.cx = []

        for i, layer in enumerate(self.hidden):
            if isinstance(layer, nn.Linear):
                self.hx.append(None)
                self.cx.append(None)
            elif isinstance(layer, nn.LSTMCell):
                self.hx.append(Variable(torch.zeros(self.batch_size, layer.hidden_size)).to(self.device))
                self.cx.append(Variable(torch.zeros(self.batch_size, layer.hidden_size)).to(self.device))
            elif isinstance(layer, nn.GRUCell):
                self.hx.append(Variable(torch.zeros(self.batch_size, layer.hidden_size)).to(self.device))
                self.cx.append(None)


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
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        policy = F.softmax(self.actor(output), dim=1)
        value = self.critic(output)
        return policy, value


class AC_RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_LSTM_layers, action_size):
        super(AC_RNN, self).__init__()
        assert type(hidden_size) == list and len(hidden_size) == 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.num_LSTM_layers = num_LSTM_layers  # number of LSTM layers
        self.lstm = torch.nn.LSTM(self.input_size, self.hidden_size[0], self.num_LSTM_layers)  # TODO: add a dropout layer?
        self.linear = torch.nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.relu = torch.nn.ReLU()
        self.actor = torch.nn.Linear(self.hidden_size[1], self.action_size)
        self.critic = torch.nn.Linear(self.hidden_size[1], 1)
        self.saved_actions = []
        self.rewards = []

    def reinit_hid(self):
        self.hidden = (torch.randn(1, self.input_size, self.hidden_size[0]),  # hx
                       torch.randn(1, self.input_size, self.hidden_size[0]))  # cx

    def forward(self, x):
        assert x.shape[-1] == self.input_size
        breakpoint()
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


def finish_trial_truncated_BPTT(model, discount_factor, optimizer, truncate_window_size, **kwargs):
    '''
    Finishes a given training trial and backpropagates, using truncated backpropagation through time (BPTT).
    '''
    assert len(model.rewards) == len(model.saved_actions)
    sequence_length = len(model.rewards)
    num_windows = sequence_length - truncate_window_size + 1
    p_losses = []
    v_losses = []

    for i_window in range(num_windows):  # roll window in forward direction

        start_idx = i_window
        end_idx = i_window + truncate_window_size

        # set the return to zero
        R = 0
        returns_ = discount_rwds(np.asarray(model.rewards[start_idx:end_idx]), gamma=discount_factor)  # [1,1,1,1] into [3.439,2.71,1.9,1]
        saved_actions = model.saved_actions[start_idx:end_idx]

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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        p_losses.append(p_loss.detach())
        v_losses.append(v_loss.detach())


    del model.rewards[:]
    del model.saved_actions[:]

    return p_losses, v_losses