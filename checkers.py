class Checker:
    def __init__(self, color, x, y):
        self.color = color
        self.is_king = False
        self.pos = (x, y)
        self.possible_moves = {}
        self.directions = ((1, -1), (-1, -1)) if color == 'B' else ((1, 1), (-1, 1))
    
    def validate_move(self, move):
        return 0 <= move[0] < 8 and 0 <= move[1] < 8
    
    def update_possible_moves(self, board, pos=None, is_capturing=False, captured_pieces=[]):
        if pos is None:
            pos = self.pos
        
        self.possible_moves = {}
        
        if self.is_king:
            self._update_king_moves(board, pos, is_capturing, captured_pieces)
        else:
            self._update_regular_moves(board, pos, is_capturing, captured_pieces)
    
    def _update_regular_moves(self, board, pos, is_capturing, captured_pieces):
        for dx, dy in self.directions:
            x, y = pos[0] + dx, pos[1] + dy
            
            if not self.validate_move((x, y)):
                continue
                
            if board[y][x] == 0 and not is_capturing:
                self.possible_moves[(x, y)] = []
            elif (board[y][x] != 0 and board[y][x].color != self.color and (x, y) not in captured_pieces):
                nx, ny = x + dx, y + dy
                if self.validate_move((nx, ny)) and board[ny][nx] == 0:
                    new_captured = captured_pieces + [(x, y)]
                    self.possible_moves[(nx, ny)] = new_captured
                    self._update_regular_moves(board, (nx, ny), True, new_captured)
    
    def _update_king_moves(self, board, pos, is_capturing, captured_pieces):
        for dx, dy in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
            found_piece = False
            x, y = pos[0] + dx, pos[1] + dy
            
            while self.validate_move((x, y)) and not found_piece:
                if board[y][x] == 0:
                    if not is_capturing:
                        self.possible_moves[(x, y)] = []
                else:
                    if (board[y][x].color != self.color and 
                        (x, y) not in captured_pieces and 
                        not is_capturing):
                        nx, ny = x + dx, y + dy
                        while self.validate_move((nx, ny)) and board[ny][nx] == 0:
                            new_captured = captured_pieces + [(x, y)]
                            self.possible_moves[(nx, ny)] = new_captured
                            self._update_king_moves(board, (nx, ny), True, new_captured)
                            nx += dx
                            ny += dy
                    found_piece = True
                
                x += dx
                y += dy
    
    def move(self, board, move_pos):
        self.update_possible_moves(board)
        
        if move_pos not in self.possible_moves:
            return False
            
        captured_positions = self.possible_moves[move_pos]
        for x, y in captured_positions:
            board[y][x] = 0
            
        old_x, old_y = self.pos
        new_x, new_y = move_pos
        board[old_y][old_x] = 0
        self.pos = move_pos
        board[new_y][new_x] = self
        
        self._check_king_promotion(new_y)
        
        return True
    
    def _check_king_promotion(self, y):
        if (y == 7 and self.color == 'W') or (y == 0 and self.color == 'B'):
            self.is_king = True
            self.directions = ((1, -1), (-1, -1), (1, 1), (-1, 1))


class Checkers:
    def __init__(self):
        self.board = [[0 for _ in range(8)] for _ in range(8)]
        self.current_player = 'W'  
        self.game_over = False
        self.winner = None
        self.move_history = []  
        self.position_count = {} 
        self.init_board()
    
    def init_board(self):
        for y in range(8):
            for x in range(8):
                if (x + y) % 2 == 1 and y < 3:
                    self.board[y][x] = Checker('W', x, y)
                elif (x + y) % 2 == 1 and y > 4:
                    self.board[y][x] = Checker('B', x, y)
    
    def print_board(self):
        print("\n   " + " ".join(f" {i} " for i in range(8)))
        for row in range(8):
            print(f" {row} ", end="")
            for col in range(8):
                piece = self.board[row][col]
                if piece == 0:
                    print("[ ]", end="")
                else:
                    king = "K" if piece.is_king else ""
                    print(f"[{piece.color}{king}]", end="")
            print()
        print()
    
    def is_valid_move(self, from_pos, to_pos):
        from_x, from_y = from_pos
        if not (0 <= from_x < 8 and 0 <= from_y < 8):
            return False
            
        checker = self.board[from_y][from_x]
        if checker == 0 or checker.color != self.current_player:
            return False
            
        checker.update_possible_moves(self.board)
        return to_pos in checker.possible_moves
    
    def get_board_state(self):
        """Возвращает строковое представление текущего состояния доски"""
        state = []
        for y in range(8):
            for x in range(8):
                piece = self.board[y][x]
                if piece == 0:
                    state.append('0')
                else:
                    state.append(f"{piece.color}{'K' if piece.is_king else 'C'}")
        return "".join(state) + self.current_player
    
    def check_game_over(self):
        """Проверяет условия окончания игры"""
        has_valid_moves = False
        white_pieces = 0
        black_pieces = 0
        
        for y in range(8):
            for x in range(8):
                piece = self.board[y][x]
                if piece != 0:
                    if piece.color == 'W':
                        white_pieces += 1
                    else:
                        black_pieces += 1
                    
                    if piece.color == self.current_player:
                        piece.update_possible_moves(self.board)
                        if piece.possible_moves:
                            has_valid_moves = True
        
        if white_pieces == 0:
            self.game_over = True
            self.winner = 'B'
            return True
            
        if black_pieces == 0:
            self.game_over = True
            self.winner = 'W'
            return True
        
        if not has_valid_moves:
            self.game_over = True
            self.winner = 'B' if self.current_player == 'W' else 'W'
            return True
        
        current_state = self.get_board_state()
        if current_state in self.position_count:
            self.position_count[current_state] += 1
            if self.position_count[current_state] >= 3:
                self.game_over = True
                self.winner = 'Draw'  
                return True
        else:
            self.position_count[current_state] = 1
        
        return False
    
    def make_move(self, from_pos, to_pos):
        from_x, from_y = from_pos
        checker = self.board[from_y][from_x]
        
        if self.is_valid_move(from_pos, to_pos) and checker.move(self.board, to_pos):
            self.move_history.append((from_pos, to_pos))
            
            if self.check_game_over():
                return True
                
            self.current_player = 'B' if self.current_player == 'W' else 'W'
            return True
        return False
    
    def game_loop(self):
        
        while not self.game_over:
            self.print_board()
            print(f"Ход игрока: {'Белые' if self.current_player == 'W' else 'Чёрные'}")
            
            try:
                move = input("Введите ход (формат: from_x from_y to_x to_y) или 'quit' для выхода: ")
                
                if move.lower() == 'quit':
                    print("Игра завершена.")
                    return
                
                coords = list(map(int, move.split()))
                if len(coords) != 4:
                    raise ValueError
                
                from_pos = (coords[0], coords[1])
                to_pos = (coords[2], coords[3])
                
                if not self.is_valid_move(from_pos, to_pos):
                    print("Недопустимый ход! Попробуйте снова.")
                    continue
                    
                if not self.make_move(from_pos, to_pos):
                    print("Ошибка при выполнении хода!")
                
            except ValueError:
                print("Некорректный ввод. Используйте формат: x1 y1 x2 y2")
        
        self.print_board()
        if self.winner == 'Draw':
            print("Игра окончена! Ничья по правилу трёх повторений позиции.")
        else:
            winner_name = "Белые" if self.winner == 'W' else "Чёрные"
            print(f"Игра окончена! Победили {winner_name}!")


if __name__ == "__main__":
    checkers = Checkers()
    checkers.game_loop()