from tensorforce.agents import DQNAgent
from coin_gym import make_gym


agent = DQNAgent(
    states_spec=dict(type='float', shape=(5,)),
    actions_spec=dict(type='int', num_actions=3),
    network_spec=[
        # dict(type='dense', size=128, activation='relu'),
        dict(type='dense', size=32, activation='relu'),
        dict(type='dense', size=32, activation='relu'),
        dict(type='dense', size=32)
    ],
    batch_size=1024,
    memory=dict(type='replay', capacity=10000)
)

MAX_STEPS = 200000

env = make_gym()
episodes = 0
while True:
    state = env.reset()
    done = False
    ave_reward = 0
    steps = 0
    while not done and steps < MAX_STEPS:
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        if done and steps < MAX_STEPS:
            reward = -100
        elif done and steps >= MAX_STEPS:
            reward = 100
        agent.observe(reward=reward, terminal=done and steps < MAX_STEPS)
        ave_reward += reward
        steps += 1
        if steps % 10000 == 0:
            print('Step %10d, Value: %.1f' % (steps, env.agent.value))
    episodes += 1
    print('\n\nEpisode %4d\n\tSteps - %d\n\tTotal reward - %.3f\n\tAverage reward - %.3f' %
          (episodes, steps, ave_reward, ave_reward / steps))
