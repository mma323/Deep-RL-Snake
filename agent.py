"""
store all the agents here
"""
from replay_buffer import ReplayBuffer, ReplayBufferNumpy
import numpy as np
import time
import pickle
from collections import deque
import json
import torch
import torch.nn as nn
import torch.optim as optim
import json


def huber_loss(y_true, y_pred, delta=1):
    """PyTorch implementation for huber loss
    loss = {
        0.5 * (y_true - y_pred)**2 if abs(y_true - y_pred) < delta
        delta * (abs(y_true - y_pred) - 0.5 * delta) otherwise
    }
    Parameters
    ----------
    y_true : Tensor
        The true values for the regression data
    y_pred : Tensor
        The predicted values for the regression data
    delta : float, optional
        The cutoff to decide whether to use quadratic or linear loss

    Returns
    -------
    loss : Tensor
        loss values for all points
    """
    error = (y_true - y_pred)
    quad_error = 0.5*torch.square(error)
    lin_error = delta*(torch.abs(error) - 0.5*delta)
    # quadratic error, linear error
    return torch.where(torch.abs(error) < delta, quad_error, lin_error)


def mean_huber_loss(y_true, y_pred, delta=1):
    """Calculates the mean value of huber loss

    Parameters
    ----------
    y_true : Tensor
        The true values for the regression data
    y_pred : Tensor
        The predicted values for the regression data
    delta : float, optional
        The cutoff to decide whether to use quadratic or linear loss

    Returns
    -------
    loss : Tensor
        average loss across points
    """
    return torch.mean(huber_loss(y_true, y_pred, delta))


class Agent():
    """Base class for all agents
    This class extends to the following classes
    DeepQLearningAgent
    HamiltonianCycleAgent
    BreadthFirstSearchAgent

    Attributes
    ----------
    _board_size : int
        Size of board, keep greater than 6 for useful learning
        should be the same as the env board size
    _n_frames : int
        Total frames to keep in history when making prediction
        should be the same as env board size
    _buffer_size : int
        Size of the buffer, how many examples to keep in memory
        should be large for DQN
    _n_actions : int
        Total actions available in the env, should be same as env
    _gamma : float
        Reward discounting to use for future rewards, useful in policy
        gradient, keep < 1 for convergence
    _use_target_net : bool
        If use a target network to calculate next state Q values,
        necessary to stabilise DQN learning
    _input_shape : tuple
        Tuple to store individual state shapes
    _board_grid : Numpy array
        A square filled with values from 0 to board size **2,
        Useful when converting between row, col and int representation
    _version : str
        model version string
    """
    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        """ initialize the agent

        Parameters
        ----------
        board_size : int, optional
            The env board size, keep > 6
        frames : int, optional
            The env frame count to keep old frames in state
        buffer_size : int, optional
            Size of the buffer, keep large for DQN
        gamma : float, optional
            Agent's discount factor, keep < 1 for convergence
        n_actions : int, optional
            Count of actions available in env
        use_target_net : bool, optional
            Whether to use target network, necessary for DQN convergence
        version : str, optional except NN based models
            path to the model architecture json
        """
        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._board_size, self._board_size, self._n_frames)
        # reset buffer also initializes the buffer
        self.reset_buffer()
        self._board_grid = np.arange(0, self._board_size**2)\
                             .reshape(self._board_size, -1)
        self._version = version


    def get_gamma(self):
        """Returns the agent's gamma value

        Returns
        -------
        _gamma : float
            Agent's gamma value
        """
        return self._gamma


    def reset_buffer(self, buffer_size=None):
        """Reset current buffer 
        
        Parameters
        ----------
        buffer_size : int, optional
            Initialize the buffer with buffer_size, if not supplied,
            use the original value
        """
        if(buffer_size is not None):
            self._buffer_size = buffer_size
        self._buffer = ReplayBufferNumpy(self._buffer_size, self._board_size, 
                                    self._n_frames, self._n_actions)


    def get_buffer_size(self):
        """Get the current buffer size
        
        Returns
        -------
        buffer size : int
            Current size of the buffer
        """
        return self._buffer.get_current_size()


    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        """Add current game step to the replay buffer

        Parameters
        ----------
        board : Numpy array
            Current state of the board, can contain multiple games
        action : Numpy array or int
            Action that was taken, can contain actions for multiple games
        reward : Numpy array or int
            Reward value(s) for the current action on current states
        next_board : Numpy array
            State obtained after executing action on current state
        done : Numpy array or int
            Binary indicator for game termination
        legal_moves : Numpy array
            Binary indicators for actions which are allowed at next states
        """
        self._buffer.add_to_buffer(board, action, reward, next_board, 
                                   done, legal_moves)


    def save_buffer(self, file_path='', iteration=None):
        """Save the buffer to disk

        Parameters
        ----------
        file_path : str, optional
            The location to save the buffer at
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'wb') as f:
            pickle.dump(self._buffer, f)


    def load_buffer(self, file_path='', iteration=None):
        """Load the buffer from disk
        
        Parameters
        ----------
        file_path : str, optional
            Disk location to fetch the buffer from
        iteration : int, optional
            Iteration number to use in case the file has been tagged
            with one, 0 if iteration is None

        Raises
        ------
        FileNotFoundError
            If the requested file could not be located on the disk
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'rb') as f:
            self._buffer = pickle.load(f)


    def _point_to_row_col(self, point):
        """Covert a point value to row, col value
        point value is the array index when it is flattened

        Parameters
        ----------
        point : int
            The point to convert

        Returns
        -------
        (row, col) : tuple
            Row and column values for the point
        """
        return (point//self._board_size, point%self._board_size)


    def _row_col_to_point(self, row, col):
        """Covert a (row, col) to value
        point value is the array index when it is flattened

        Parameters
        ----------
        row : int
            The row number in array
        col : int
            The column number in array
        Returns
        -------
        point : int
            point value corresponding to the row and col values
        """
        return row*self._board_size + col
    

class DeepQLearningAgent(Agent):
    """This agent learns the game via Q learning
    model outputs everywhere refers to Q values
    This class extends to the following classes
    PolicyGradientAgent
    AdvantageActorCriticAgent

    Attributes
    ----------
    _model : PyTorch Graph
        Stores the graph of the DQN model
    _target_net : PyTorch Graph
        Stores the target network graph of the DQN model
    """
    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        """Initializer for DQN agent, arguments are same as Agent class
        except use_target_net is by default True and we call and additional
        reset models method to initialize the DQN networks
        """
        Agent.__init__(self, board_size=board_size, frames=frames, buffer_size=buffer_size,
                 gamma=gamma, n_actions=n_actions, use_target_net=use_target_net,
                 version=version)
        self.reset_models()


    def reset_models(self):
        """ Reset all the models by creating new graphs"""
        self._model = self._agent_model()
        if(self._use_target_net):
            self._target_net = self._agent_model()
            self.update_target_net()


    def _prepare_input(self, board):
        """Reshape input and normalize
        
        Parameters
        ----------
        board : Numpy array
            The board state to process

        Returns
        -------
        board : Numpy array
            Processed and normalized board
        """
        if(board.ndim == 3):
            board = board.reshape((1,) + self._input_shape)
        board = self._normalize_board(board.copy())
        return board.copy()


    def _get_model_outputs(self, board, model=None):
            """Get action values from the DQN model

            Parameters
            ----------
            board : Numpy array
                The board state for which to predict action values
            model : PyTorch Graph, optional
                The graph to use for prediction, model or target network

            Returns
            -------
            model_outputs : Numpy array
                Predicted model outputs on board, 
                of shape board.shape[0] * num actions
            """
            board = self._prepare_input(board)
            if model is None:
                model = self._model
            with torch.no_grad():
                model_outputs = model(torch.from_numpy(board))
            return model_outputs.numpy()
    
    
    def _normalize_board(self, board):
        """Normalize the board before input to the network
        
        Parameters
        ----------
        board : Numpy array
            The board state to normalize

        Returns
        -------
        board : Numpy array
            The copy of board state after normalization
        """
        # return board.copy()
        # return((board/128.0 - 1).copy())
        return board.astype(np.float32)/4.0


    def move(self, board, legal_moves, value=None):
        """Get the action with maximum Q value
        
        Parameters
        ----------
        board : Numpy array
            The board state on which to calculate best action
        value : None, optional
            Kept for consistency with other agent classes

        Returns
        -------
        output : Numpy array
            Selected action using the argmax function
        """
        model_outputs = self._get_model_outputs(board, self._model)
        return np.argmax(np.where(legal_moves==1, model_outputs, -np.inf), axis=1)
    

    class AgentModel(nn.Module):
        def __init__(self, board_size, frames, n_actions, version):
            super().__init__()
            self.board_size = board_size
            self.frames = frames
            self.n_actions = n_actions
            self.version = version

            with open('model_config/{:s}.json'.format(self.version), 'r') as f:
                model_config = json.loads(f.read())

            in_channels = frames
            output_size = np.array(board_size)  
            layers = []

            for l in model_config:
                if 'Conv2D' in l:
                    conv_layer = nn.Conv2d(in_channels, l['filters'], tuple(l['kernel_size']), stride=tuple(l['strides']))
                    layers.append(conv_layer)
                    layers.append(nn.ReLU())
                    # In order to to match output size of Conv2D
                    in_channels = l['filters']
                    # In order to match output size of Conv2D
                    output_size = ((output_size - np.array(l['kernel_size'])) // np.array(l['strides'])) + 1
                elif 'Flatten' in l:
                    layers.append(nn.Flatten())
                    # Done since Flatten reduces dimensions
                    in_channels = 1
                    output_size = 1  # Done since Flatten reduces dimensions
                elif 'Dense' in l:
                    layers.append(nn.Linear(in_channels * output_size * output_size, l['units']))
                    layers.append(nn.ReLU())
                    in_channels = l['units']  #Done to match output size of Dense

            layers.append(nn.Linear(in_channels * output_size * output_size, self.n_actions))
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            x = x.permute(0, 3, 1, 2)  # To match what Conv2D expects
            x = x.reshape(x.size(0), -1)  # To match what Flatten expects
            return self.model(x)


    def _agent_model(self):
        """Returns the model which evaluates Q values for a given state input

        Returns
        -------
        model : PyTorch Graph
            DQN model graph
        """
        return self.AgentModel(self._board_size, self._n_frames, self._n_actions, self._version)


    def set_loss_and_optimizer(self):
        self.optimizer = optim.RMSprop(self._model.parameters(), lr=0.0005)
        self.loss = mean_huber_loss


    def get_action_proba(self, board, values=None):
        """Returns the action probability values using the DQN model

        Parameters
        ----------
        board : Numpy array
            Board state on which to calculate action probabilities
        values : None, optional
            Kept for consistency with other agent classes
        
        Returns
        -------
        model_outputs : Numpy array
            Action probabilities, shape is board.shape[0] * n_actions
        """
        model_outputs = self._get_model_outputs(board, self._model)
        # subtracting max and taking softmax does not change output
        # do this for numerical stability
        model_outputs = np.clip(model_outputs, -10, 10)
        model_outputs = model_outputs - model_outputs.max(axis=1).reshape((-1,1))
        model_outputs = np.exp(model_outputs)
        model_outputs = model_outputs/model_outputs.sum(axis=1).reshape((-1,1))
        return model_outputs

    
    def save_model(self, file_path='', iteration=None):
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        torch.save(self._model.state_dict(), "{}/model_{:04d}.pth".format(file_path, iteration))
        if(self._use_target_net):
            torch.save(self._target_net.state_dict(), "{}/model_{:04d}_target.pth".format(file_path, iteration))


    def load_model(self, file_path='', iteration=None):
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._model.load_state_dict(torch.load("{}/model_{:04d}.pth".format(file_path, iteration)))
        if(self._use_target_net):
            self._target_net.load_state_dict(torch.load("{}/model_{:04d}_target.pth".format(file_path, iteration)))
        # print("Couldn't locate models at {}, check provided path".format(file_path))


    def print_models(self):
        print(self._model)


    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        """Train the agent using the replay buffer

        Parameters
        ----------
        batch_size : int, optional
            Batch size to use for training
        num_games : int, optional
            Number of games to sample from the buffer
        reward_clip : bool, optional
            Whether to clip the rewards or not

        Returns
        -------
        loss : float
            Average loss over all the batches
        """


        boards, actions, rewards, next_boards, dones, legal_moves = self._buffer.sample()

        model_outputs = self._get_model_outputs(boards, self._model)
        next_model_outputs = self._get_model_outputs(next_boards, self._target_net)
        target_values = rewards + self._gamma * np.max(next_model_outputs, axis=1) * (1-dones)
        if(reward_clip):
            target_values = np.clip(target_values, -1, 1)
        actions_indices = np.argmax(actions, axis=1)

        model_outputs = model_outputs[np.arange(model_outputs.shape[0]), actions_indices]

        target_values_tensor = torch.from_numpy(target_values).float().requires_grad_(True)
        model_outputs_tensor = torch.from_numpy(model_outputs).float().requires_grad_(True)

        self.set_loss_and_optimizer()
        loss = self.loss(target_values_tensor, model_outputs_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


    def update_target_net(self):
        """Update the weights of the target network, which is kept
        static for a few iterations to stabilize the other network.
        This should not be updated very frequently
        """
        if(self._use_target_net):
            self._target_net.load_state_dict(self._model.state_dict())


    def compare_weights(self):
        """Simple utility function to check if the model and target 
        network have the same weights or not
        """
        for i, (model_param, target_param) in enumerate(zip(self._model.parameters(), self._target_net.parameters())):
            c = torch.equal(model_param.data, target_param.data)
            print('Layer {:d} Weights Match: {:d}'.format(i, int(c)))


    def copy_weights_from_agent(self, agent_for_copy):
        """Update weights between competing agents which can be used
        in parallel training
        """
        assert isinstance(agent_for_copy, self), "Agent type is required for copy"

        self._model.load_state_dict(agent_for_copy._model.state_dict())
        self._target_net.load_state_dict(agent_for_copy._model.state_dict())