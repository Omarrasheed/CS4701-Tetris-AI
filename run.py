import random
from datetime import datetime
from statistics import mean, median

from tqdm import tqdm

from dqn_algorithm import DQNAlgorithm
from heuristic_algorithm import HeuristicAlgorithm
from logs import CustomTensorBoard
from tetris import Tetris


def heuristic():
    env = Tetris()
    algo = HeuristicAlgorithm()

    current_state = env.reset()
    done = False
    steps = 0

    while not done:
        next_states = env.get_next_states()
        best_state = algo.best_state(next_states.values())
        
        best_action = None
        for action, state in next_states.items():
            if state == best_state:
                best_action = action
                break

        reward, done = env.play(best_action[0], best_action[1], render=True,
                                render_delay=0)
        
        algo.add_to_memory(current_state, next_states[best_action], reward, done)
        current_state = next_states[best_action]
        steps += 1
    

# Run dqn with Tetris
def dqn():
    env = Tetris()
    episodes = 4000
    max_steps = None
    batch_size = 512
    epochs = 1
    render_every = 50
    log_every = 50
    replay_start_size = 2000
    train_every = 1
    render_delay = None

    algo = DQNAlgorithm(env.get_state_size())

    log_dir = f'logs/tetris-nn={str([32, 32])}-mem={20000}-bs={batch_size}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    log = CustomTensorBoard(log_dir=log_dir)

    scores = []
    times = []

    for episode in tqdm(range(episodes)):
        current_state = env.reset()
        done = False
        steps = 0

        if render_every and episode % render_every == 0:
            render = True
        else:
            render = False

        # Game
        while not done and (not max_steps or steps < max_steps):
            next_states = env.get_next_states()
            best_state = algo.best_state(next_states.values())
            
            best_action = None
            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
                    break

            reward, done = env.play(best_action[0], best_action[1], render=render,
                                    render_delay=render_delay)
            
            algo.add_to_memory(current_state, next_states[best_action], reward, done)
            current_state = next_states[best_action]
            steps += 1

        scores.append(env.get_game_score())
        times.append(env.get_game_time())


        # Train
        if episode % train_every == 0:
            algo.train(batch_size=batch_size, epochs=epochs)

        # Logs
        if log_every and episode and episode % log_every == 0:
            score = scores[-log_every:]
            time = times[-log_every:]
            avg_score = mean(score)
            min_score = min(score)
            max_score = max(score)

            avg_time = mean(time)
            min_time = min(time)
            max_time = max(time)

            log.log(episode, avg_score=avg_score, min_score=min_score, max_score=max_score, avg_time=avg_time, min_time=min_time, max_time=max_time)


if __name__ == "__main__":
    # dqn()
    times = 0
    for i in range(20):
        now = datetime.now()
        heuristic()
        times += (datetime.now() - now).total_seconds()
    print(times / 5.0)
