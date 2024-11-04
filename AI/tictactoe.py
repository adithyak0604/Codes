board=[['' for _ in range(3)] for _ in range(3)]
def print_board():
    for row in board:
        print(' | '.join(row))
        print("---------")

def check_win(player):
    for row in board:
        if row.count(player)==3:
            return True
        for col in range(3):
            if all(board[i][col]==player for i in range(3)):
                return True
        if all(board[i][i]==player for i in range(3)) or all(board[i][2-i]==player for i in range(3)):
            return True
        return False

def game():
    current_player='X'
    while True:
        print_board()
        print("Make Move ",current_player)
        row=int(input("Enter row(0-2): "))
        col=int(input("Enter column(0-2):"))
        if board[row][col]=='':
            board[row][col]=current_player
            if check_win(current_player):
                print_board()
                print(f"Player {current_player} wins!...")
                break
            current_player='O' if current_player=='X' else 'X'
            if all(cell!='' for row in board for cell in row):
                print_board()
                print("It's a Draw!")
                break
        else:
            print("Invalid Move, try again")

game()