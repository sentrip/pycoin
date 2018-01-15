import random
from pycoin.envs import *
from pycoin.gens import *
from pycoin.utils import strategy_performance

if __name__ == '__main__':
    env_type = 'T'  # T - Trading, L - LiveTrading, ANYTHING_ELSE - BackTest
    action_type = 'input'  # input - manual input, ANYTHING_ELSE - random actions
    data_type = 'w'  # w - wavy_signal, s - step_signal, ANYTHING_ELSE - historical

    if env_type == 'L':
        env = LiveTrading('btc',
                          history_length=4, step_length=1,
                          fee=0.1, max_position=3)
    else:
        data = wavy_signal(50) if data_type == 'w' \
            else step_signal(50) if data_type == 's' \
            else historical('btc', period='h')

        if env_type == 'T':
            env = Trading(data,
                          history_length=4, step_length=1,
                          fee=0.1, max_position=5)
        else:
            env = BackTest(data,
                           history_length=4, step_length=1,
                           fee=0.1, max_position=5)

    while True:
        env.reset()
        if env_type in ['T', 'L']:
            env.render()
        done = False
        info = []
        while not done:
            ct = input() if action_type == 'input' else ('' if random.random() < 0.9 else random.choice(['b', 's']))
            st, r, done, info = env.step(1 if ct == 'b' else 2 if ct == 's' else 0)
            print(st, r)
            print('-----------------------------------')
            if env_type in ['T', 'L']:
                env.render()
        if env_type not in ['T', 'L']:
            details = pd.DataFrame(info)
            strategy_performance(details=details)
