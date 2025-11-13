import pygame
import sys
import numpy as np
from neural_network import NN
from checkers import Checkers
from copy import deepcopy

class CheckersUI:
    def __init__(self, model_path1=None, model_path2=None):
        pygame.init()
        self.WIDTH, self.HEIGHT = 600, 600
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Checkers AI")
        
        self.cell_size = self.WIDTH // 8
        self.colors = {
            'light': (240, 217, 181),
            'dark': (181, 136, 99),
            'highlight': (106, 190, 48),
            'text': (0, 0, 0)
        }
        
        self.game = Checkers()
        self.ai1 = CheckersAI(model_path1, 'AI-1')
        self.ai2 = CheckersAI(model_path2, 'AI-2')
        self.running = True
        self.ai_thinking = False
        self.move_delay = 600 
        self.last_move_time = 0
        
    def draw_board(self):
        for row in range(8):
            for col in range(8):
                color = self.colors['light'] if (row + col) % 2 == 0 else self.colors['dark']
                pygame.draw.rect(self.screen, color, 
                               (col * self.cell_size, row * self.cell_size, 
                                self.cell_size, self.cell_size))
    
    def draw_pieces(self):
        for row in range(8):
            for col in range(8):
                piece = self.game.board[row][col]
                if piece != 0:
                    x = col * self.cell_size + self.cell_size // 2
                    y = row * self.cell_size + self.cell_size // 2
                    radius = self.cell_size // 2 - 10
                    
                    color = (255, 255, 255) if piece.color == 'W' else (50, 50, 50)
                    pygame.draw.circle(self.screen, color, (x, y), radius)
                    pygame.draw.circle(self.screen, (100, 100, 100), (x, y), radius, 2)
                    
                    if piece.is_king:
                        crown_color = (255, 215, 0) if piece.color == 'W' else (200, 150, 0)
                        pygame.draw.circle(self.screen, crown_color, (x, y), radius // 2)
    
    def draw_info(self):
        font = pygame.font.SysFont('Arial', 20)
        
        current_ai = self.ai1 if self.game.current_player == 'W' else self.ai2
        player_text = f"Current: {current_ai.name}"
        text_surface = font.render(player_text, True, self.colors['text'])
        self.screen.blit(text_surface, (10, self.HEIGHT - 60))
        
        white_count = sum(1 for row in self.game.board for piece in row if piece != 0 and piece.color == 'W')
        black_count = sum(1 for row in self.game.board for piece in row if piece != 0 and piece.color == 'B')
        count_text = f"White: {white_count}  Black: {black_count}"
        text_surface = font.render(count_text, True, self.colors['text'])
        self.screen.blit(text_surface, (10, self.HEIGHT - 30))
        
        if self.ai_thinking:
            thinking_text = "AI Thinking..."
            text_surface = font.render(thinking_text, True, (200, 0, 0))
            self.screen.blit(text_surface, (self.WIDTH - 150, self.HEIGHT - 30))
    
    def ai_move(self):
        current_ai = self.ai1 if self.game.current_player == 'W' else self.ai2
        move = current_ai.choose_action(self.game, self.game.current_player, training=False)
        
        if move:
            from_pos, to_pos = move
            self.game.make_move(from_pos, to_pos)
            
            if current_ai.is_game_over(self.game):
                winner = "White" if current_ai.is_winning_position(self.game, 'W') else "Black"
                print(f"Game Over! Winner: {winner}")
                self.running = False
    
    def run(self):
        clock = pygame.time.Clock()
        
        while self.running:
            current_time = pygame.time.get_ticks()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        self.ai_thinking = True
                        self.ai_move()
                        self.ai_thinking = False
            
            if not self.ai_thinking and current_time - self.last_move_time > self.move_delay:
                self.ai_thinking = True
                self.ai_move()
                self.last_move_time = current_time
                self.ai_thinking = False
            
            self.draw_board()
            self.draw_pieces()
            self.draw_info()
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
        sys.exit()

class CheckersAI:
    def __init__(self, model_path=None, name="AI"):
        self.model = NN(128, 64, 32, 1)
        self.name = name
        if model_path:
            try:
                self.model.load()
            except:
                pass
    
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
        return np.array(norm_board).reshape(1, -1)
    
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
    
    def choose_action(self, checkers_game, player_color, training=False):
        legal_moves = self.get_legal_moves(checkers_game, player_color)
        if not legal_moves:
            return None
        
        best_move = None
        best_value = -np.inf
        
        for (from_pos, to_pos) in legal_moves:
            temp_game = self.simulate_move(checkers_game, from_pos, to_pos)
            next_state = self.board_to_features(temp_game.board)
            q_value = self.model.forward(next_state)[0][0]
            
            if q_value > best_value:
                best_value = q_value
                best_move = (from_pos, to_pos)
        
        return best_move
    
    def simulate_move(self, checkers_game, from_pos, to_pos):
        temp_game = deepcopy(checkers_game)
        temp_game.make_move(from_pos, to_pos)
        return temp_game
    
    def is_winning_position(self, checkers_game, player_color):
        if checkers_game.check_game_over():
            return player_color == checkers_game.winner if checkers_game.winner != 'Draw' else player_color != checkers_game.winner

    
    def is_game_over(self, checkers_game):
        return (self.is_winning_position(checkers_game, 'W') or 
                self.is_winning_position(checkers_game, 'B'))

if __name__ == "__main__":
    ui = CheckersUI('checkers_model.json', 'checkers_model.json')
    ui.run()