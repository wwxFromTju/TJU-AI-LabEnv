import matplotlib
# matplotlib.use('TkAgg')  # avoid non-GUI warning for matplotlib

from IPython.display import display, HTML

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class RandomMove(gym.Env):
    def __init__(self, len=7, end_reward=[0, 1], live_display=False):
        self.n = len
        self.start_idx = len // 2
        self.idx = self.start_idx
        self.reward_in = end_reward
        self.end_node = [0, self.n - 1]
        self.end_reward = end_reward

        self.action_set = [0, 1]
        self.action_eff = [-1, 1]

        self.cmap = colors.ListedColormap(['white', 'black', 'blue', 'green', 'red', 'gray'])
        self.bounds = [0, 1, 2, 3, 4, 5, 6] 
        self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)
        
        self.ax_imgs = []
        
        self.live_display = live_display

        
    def step(self, action):
        if self.idx in self.end_node:
            print('game is end! please reset')
            return self.n, 0, True, {}
        
        if action not in self.action_set:
            print('use action in {}'.format(self.action_set))
            return self.idx, 0, False, {}
        
        reward = 0
        done = False
        self.idx += self.action_eff[action]
        if self.idx in self.end_node:
            done = True
            reward = self.end_reward[self.end_node.index(self.idx)]                
        return self.idx, reward, done, {}

    def reset(self):
        self.idx = self.start_idx
        self.ax_imgs = []
        return self.idx

    def _obs(self):
        obs = np.zeros(self.n).reshape(1, -1) + 1
        
        obs[0][self.n - 1] =2
        obs[0][self.idx] = 3

        return obs
    
    def render(self, mode='human', close=False):
        if close:
            plt.close()
            return
        
        obs = self._obs()

        if not hasattr(self, 'fig'):  
            self.fig, self.ax_full= plt.subplots(nrows=1, ncols=1)
        self.ax_full.axis('off')
        
        self.fig.show()
        if self.live_display:
            if not hasattr(self, 'ax_full_img'):
                self.ax_full_img = self.ax_full.imshow(obs, cmap=self.cmap, norm=self.norm, animated=True)
            self.ax_full_img.set_data(obs)
        else:
            self.ax_full_img = self.ax_full.imshow(obs, cmap=self.cmap, norm=self.norm, animated=True)
           
        plt.draw()
        
        if self.live_display:
            self.fig.canvas.draw()
        else:
            self.ax_imgs.append([self.ax_full_img])  # List of axes to update figure frame
            self.fig.set_dpi(100)

        return self.fig
    
    def _get_video(self, interval=200, gif_path=None):
        if self.live_display:
            # TODO: Find a way to create animations without slowing down the live display
            print('Warning: Generating an Animation when live_display=True not yet supported.')
        anim = animation.ArtistAnimation(self.fig, self.ax_imgs, interval=interval)
        
        if gif_path is not None:
            anim.save(gif_path, writer='imagemagick', fps=10)
        return anim