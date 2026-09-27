







from importlib import import_module
import numpy as np









class Environment:
    def __init__(self, env_name, sticky_action_prob = 0.1, difficulty_ramping = False, random_seed = None, use_minimal_observation=True):
        env_module = import_module(env_name)
        self.random = np.random.RandomState(random_seed)
        self.env_name = env_name
        self.env = env_module.Env(ramping = difficulty_ramping, random_state = self.random, use_minimal_observation=use_minimal_observation)
        self.n_channels = self.env.state_shape()[2]
        self.sticky_action_prob = sticky_action_prob
        self.last_action = 0
        self.visualized = False
        self.closed = False


    def act(self, a):
        if(self.random.rand()<self.sticky_action_prob):
            a = self.last_action
        self.last_action = a
        return self.env.act(a)


    def state(self):
        return self.env.state()


    def reset(self):
        return self.env.reset()


    def state_shape(self):
        return self.env.state_shape()


    def num_actions(self):
        return 6


    def game_name(self):
        return self.env_name


    def minimal_action_set(self):
        return self.env.minimal_action_set()


    def display_state(self, time=50):
        if(not self.visualized):
            global plt
            global colors
            global sns
            mpl = __import__('matplotlib.pyplot', globals(), locals())
            plt = mpl.pyplot
            mpl = __import__('matplotlib.colors', globals(), locals())
            colors = mpl.colors
            sns = __import__('seaborn', globals(), locals())
            self.cmap = sns.color_palette("cubehelix", self.n_channels)
            self.cmap.insert(0,(0,0,0))
            self.cmap=colors.ListedColormap(self.cmap)
            bounds = [i for i in range(self.n_channels+2)]
            self.norm = colors.BoundaryNorm(bounds, self.n_channels+1)
            _, self.ax = plt.subplots(1,1)
            plt.show(block=False)
            self.visualized = True
        if(self.closed):
            _, self.ax = plt.subplots(1,1)
            plt.show(block=False)
            self.closed = False
        state = self.env.state()
        numerical_state = np.amax(state*np.reshape(np.arange(self.n_channels)+1,(1,1,-1)),2)+0.5
        self.ax.imshow(numerical_state, cmap=self.cmap, norm=self.norm, interpolation='none')
        plt.pause(time/1000)
        plt.cla()

    def close_display(self):
        plt.close()
        self.closed = True
