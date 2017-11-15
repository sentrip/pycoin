from coin_gym import make_gym
from tests import connect_and_send
from threading import Thread

if __name__ == '__main__':
    Thread(target=connect_and_send, kwargs={'ln': 10000}).start()
    env = make_gym()
    while True:
        print('\n\n\nResetting Environment...\n\n\n')
        env.reset()
        done = False
        while not done:
            _in = input('Action, n_times: ')
            try:
                action, n = _in.split(',')
                action, n = int(action), int(n)
            except ValueError:
                try:
                    action, n = int(_in), 1
                except ValueError:
                    action, n = 1,  1
            if n == 1:
                state, reward, done, _ = env.step(action)
                print('Reward: {:.3f}, Value: {:7d}, Price: {:.2f}'.format(reward, int(env.agent.value), env.market.price))
            else:
                rewards = []
                states = []
                for _ in range(n):
                    s, r, done, _ = env.step(action)
                    rewards.append(r)
                    states.append(s)
                    if done:
                        break
                reward = sum(rewards) / max(len(rewards), 1)
                mx, mn = max(rewards), min(rewards)
                print('Reward: {:.3f}, Value: {:7d}, Price: {:.1f}, '
                      'Max reward: {:.3f}, Min reward: {:.3f}'.format(reward, int(env.agent.value),
                                                                      env.market.price, mx, mn))
