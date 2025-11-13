from neural_network import NN 
from checkers import Checkers
import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class neurocheckers:
    def __init__(self, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.game = Checkers()
        self.model = NN(128, 64, 32, 1)  
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.replay_buffer = ReplayBuffer(10000)
        self.batch_size = 32
        self.moves_without_capture = 0

    def board_to_features(self, board):
        norm_board = []
        for y in range(8):
            for x in range(8):
                if (x + y) % 2 == 1: 
                    piece = board[y][x]
                    if piece == 0:
                        norm_board.extend([0, 0, 0, 0])
                    else:
                        is_king = 1 if piece.is_king else 0
                        is_white = 1 if piece.color == 'W' else 0
                        is_black = 1 if piece.color == 'B' else 0
                        norm_board.extend([1, is_king, is_white, is_black])
        
        features = np.array(norm_board).reshape(1, -1)
        return features
    
    def get_legal_moves(self, checkers_game, player_color):
        legal_moves = []
        
        for y in range(8):
            for x in range(8):
                piece = checkers_game.board[y][x]
                if piece != 0 and piece.color == player_color:
                    piece.update_possible_moves(checkers_game.board)
                    for move_pos in piece.possible_moves:
                        legal_moves.append(((x, y), move_pos))
        
        return legal_moves

    def choose_action(self, checkers_game, player_color, training=True):
        legal_moves = self.get_legal_moves(checkers_game, player_color)
        
        if not legal_moves:
            return None
        
        if training and np.random.random() < self.epsilon:
            return random.choice(legal_moves)
        
        best_move = None
        best_value = -np.inf
        
        for (from_pos, to_pos) in legal_moves:
            temp_game = self.simulate_move(checkers_game, from_pos, to_pos)
            next_state = self.board_to_features(temp_game.board)
            
            q_value = self.model.forward(next_state)
            q_value = q_value[0][0]  
            
            if q_value > best_value:
                best_value = q_value
                best_move = (from_pos, to_pos)
        
        return best_move
    
    def simulate_move(self, checkers_game, from_pos, to_pos):
        temp_game = deepcopy(checkers_game)
        temp_game.make_move(from_pos, to_pos)
        return temp_game
    
    def get_piece_count(self, checkers_game, player_color):
        count = 0
        for y in range(8):
            for x in range(8):
                piece = checkers_game.board[y][x]
                if piece != 0 and piece.color == player_color:
                    count += 1
        return count

    def get_kings_count(self, checkers_game, player_color):
        kings = 0
        for y in range(8):
            for x in range(8):
                piece = checkers_game.board[y][x]
                if piece != 0 and piece.color == player_color and piece.is_king:
                    kings += 1
        return kings

    def is_winning_position(self, checkers_game, player_color):
        opponent_color = 'B' if player_color == 'W' else 'W'
        opponent_pieces = self.get_piece_count(checkers_game, opponent_color)
        if opponent_pieces == 0:
            return True
        opponent_moves = self.get_legal_moves(checkers_game, opponent_color)
        return len(opponent_moves) == 0

    def is_losing_position(self, checkers_game, player_color):
        player_pieces = self.get_piece_count(checkers_game, player_color)
        if player_pieces == 0:
            return True
        player_moves = self.get_legal_moves(checkers_game, player_color)
        return len(player_moves) == 0

    def is_draw(self, checkers_game, max_moves_without_capture=30):
        return self.moves_without_capture >= max_moves_without_capture
    
    def is_game_over(self, checkers_game):
        return (self.is_winning_position(checkers_game, 'W') or 
                self.is_winning_position(checkers_game, 'B') or
                self.is_draw(checkers_game))

    def calculate_reward(self, checkers_game, player_color, move_made, prev_state=None):
        if move_made is None:
            return -50
        
        reward = 0
        opponent_color = 'B' if player_color == 'W' else 'W'
        
        player_pieces = self.get_piece_count(checkers_game, player_color)
        opponent_pieces = self.get_piece_count(checkers_game, opponent_color)
        piece_diff = player_pieces - opponent_pieces
        reward += piece_diff * 3
        
        kings_count = self.get_kings_count(checkers_game, player_color)
        reward += kings_count * 2
        
        mobility = len(self.get_legal_moves(checkers_game, player_color))
        reward += mobility * 0.2
        
        if prev_state:
            prev_opponent_pieces = self.get_piece_count(prev_state, opponent_color)
            captures = prev_opponent_pieces - opponent_pieces
            reward += captures * 10
            
            if captures > 0:
                self.moves_without_capture = 0
            else:
                self.moves_without_capture += 1
        
        if self.is_winning_position(checkers_game, player_color):
            reward += 200
        elif self.is_losing_position(checkers_game, player_color):
            reward -= 200
        elif self.is_draw(checkers_game):
            reward -= 10
        
        return reward

    def calculate_target(self, reward, next_state, done):
        if done:
            return reward
        else:
            next_q = self.model.forward(next_state)[0][0]
            return reward + self.gamma * next_q

    def train_step(self, state, reward, next_state, done):
        target = self.calculate_target(reward, next_state, done)
        
        current_q = self.model.forward(state)[0][0]
        
        error = target - current_q
        
        self.model.backward(state, np.array([[error]]))

    def train_from_buffer(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = self.replay_buffer.sample(self.batch_size)
        
        for state, _, reward, next_state, done in batch:
            self.train_step(state, reward, next_state, done)

    def evaluate_model(self, num_games=100):
        wins = 0
        draws = 0
        losses = 0
        
        for game_num in range(num_games):
            game = Checkers()
            self.moves_without_capture = 0
            
            while True:
                state = self.board_to_features(game.board)
                action = self.choose_action(game, 'W', training=False)
                
                if action is None:
                    draws += 1
                    break
                    
                prev_game_state = deepcopy(game)
                game.make_move(*action)
                
                opponent_color = 'B'
                prev_opponent_pieces = self.get_piece_count(prev_game_state, opponent_color)
                current_opponent_pieces = self.get_piece_count(game, opponent_color)
                if prev_opponent_pieces - current_opponent_pieces > 0:
                    self.moves_without_capture = 0
                else:
                    self.moves_without_capture += 1
                
                if self.is_game_over(game):
                    if self.is_winning_position(game, 'W'):
                        wins += 1
                    elif self.is_winning_position(game, 'B'):
                        losses += 1
                    else:
                        draws += 1
                    break
                
                black_moves = self.get_legal_moves(game, 'B')
                if not black_moves:
                    wins += 1
                    break
                    
                black_action = random.choice(black_moves)
                prev_game_state_black = deepcopy(game)
                game.make_move(*black_action)
                
                white_color = 'W'
                prev_white_pieces = self.get_piece_count(prev_game_state_black, white_color)
                current_white_pieces = self.get_piece_count(game, white_color)
                if prev_white_pieces - current_white_pieces > 0:
                    self.moves_without_capture = 0
                else:
                    self.moves_without_capture += 1
                
                if self.is_game_over(game):
                    if self.is_winning_position(game, 'W'):
                        wins += 1
                    elif self.is_winning_position(game, 'B'):
                        losses += 1
                    else:
                        draws += 1
                    break
        
        win_rate = wins / num_games
        print(f"Win rate: {win_rate:.3f} (W: {wins}, L: {losses}, D: {draws})")
        return win_rate

    def plot_training_progress(self, rewards, win_rates):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(rewards)
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        
        plt.subplot(1, 2, 2)
        plt.plot(win_rates)
        plt.title('Win Rate')
        plt.xlabel('Evaluation Step')
        plt.ylabel('Win Rate')
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.show()

    def train_ai(self, episodes=1000):
        print("Начало обучения...")
        best_win_rate = 0
        all_rewards = []
        all_win_rates = []
        
        for episode in range(episodes):
            game = Checkers()
            total_reward = 0
            steps = 0
            prev_state = None
            self.moves_without_capture = 0
            
            while True:
                state = self.board_to_features(game.board)
                action = self.choose_action(game, 'W', training=True)
                
                if action is None:
                    break
                
                from_pos, to_pos = action
                prev_game_state = deepcopy(game)
                game.make_move(from_pos, to_pos)
                
                opponent_color = 'B'
                prev_opponent_pieces = self.get_piece_count(prev_game_state, opponent_color)
                current_opponent_pieces = self.get_piece_count(game, opponent_color)
                if prev_opponent_pieces - current_opponent_pieces > 0:
                    self.moves_without_capture = 0
                else:
                    self.moves_without_capture += 1
                
                reward = self.calculate_reward(game, 'W', action, prev_game_state)
                total_reward += reward
                
                next_state = self.board_to_features(game.board)
                done = self.is_game_over(game)
                
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                if not done:
                    black_moves = self.get_legal_moves(game, 'B')
                    if black_moves:
                        black_action = random.choice(black_moves)
                        prev_game_state_black = deepcopy(game)
                        game.make_move(*black_action)
                        
                        white_color = 'W'
                        prev_white_pieces = self.get_piece_count(prev_game_state_black, white_color)
                        current_white_pieces = self.get_piece_count(game, white_color)
                        if prev_white_pieces - current_white_pieces > 0:
                            self.moves_without_capture = 0
                        else:
                            self.moves_without_capture += 1
                    else:
                        done = True
                        reward += 100
                        total_reward += 100
                
                self.train_from_buffer()
                
                steps += 1
                if done or steps > 200:
                    break
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            all_rewards.append(total_reward)
            
            if episode % 50 == 0:
                win_rate = self.evaluate_model(20)
                all_win_rates.append(win_rate)
                print(f"Episode {episode}, Reward: {total_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}, Win rate: {win_rate:.3f}")
                
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    self.model.save('best_model.pkl')
                    print(f"Новая лучшая модель сохранена! Win rate: {win_rate:.3f}")
            
            if episode % 200 == 0:
                self.model.save(f'model_episode_{episode}.pkl')
        
        self.plot_training_progress(all_rewards, all_win_rates)
        print(f"Обучение завершено! Лучший win rate: {best_win_rate:.3f}")

if __name__ == "__main__":
    nc = neurocheckers()
    try:
        nc.model.load()
        print("Модель загружена успешно")
    except:
        print("Начата обучение с новой моделью")
    
    nc.train_ai(episodes=10000)