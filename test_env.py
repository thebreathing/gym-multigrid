import gymnasium as gym
import time
from gymnasium import register
import argparse

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-e', '--env', default='soccer', type=str)

args = parser.parse_args()

def main():

    register(
        id='multigrid-Soccer-v0',
        entry_point='gym_multigrid.envs:SoccerGame4HEnv10x15N2',
    )
    env = gym.make('multigrid-Soccer-v0',render_mode="human")


    _ = env.reset()

    nb_agents = len(env.agents)

    while True:
        startTime= time.time()
        env.render()
        ac = [env.action_space.sample() for _ in range(nb_agents)]

        obs, _, done, _, _ = env.step(ac)
        if done:
            break

if __name__ == "__main__":
    main()