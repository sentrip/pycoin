from tensorforce.agents import PPOAgent
from coin_gym import make_gym


# Create a Proximal Policy Optimization agent
agent = PPOAgent(
    states_spec=dict(type='float', shape=(9,)),
    actions_spec=dict(type='int', num_actions=3),
    network_spec=[
        dict(type='dense', size=64),
        dict(type='dense', size=64)
    ],
    batch_size=1000,
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-4
    )
)


env = make_gym()
episodes = 0
while True:
    state = env.reset()
    done = False
    ave_reward = 0
    steps = 0
    while not done:
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        agent.observe(reward=reward, terminal=done)
        ave_reward += reward
        steps += 1
    episodes += 1
    print('\n\nEpisode %4d\n\tSteps - %d\n\tTotal reward - %.3f\n\tAverage reward - %.3f' %
          (episodes, steps, ave_reward, ave_reward / steps))
