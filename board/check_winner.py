# https://stackoverflow.com/questions/21641807/python-connect-4?answertab=active#tab-top

def check_winner(observation):
    """
    This function return the value of the winner.

    INPUT:  observation
    OUTPUT: 1 for user Winner or 2 for Computer Winner
    """

    line1 = observation.board[0:7]  # bottom row
    line2 = observation.board[7:14]
    line3 = observation.board[14:21]
    line4 = observation.board[21:28]
    line5 = observation.board[28:35]
    line6 = observation.board[35:42]

    board = [line1, line2, line3, line4, line5, line6]

    # Check rows for winner
    for row in range(6):
        for col in range(4):
            if (board[row][col] == board[row][col + 1] == board[row][col + 2] == \
                board[row][col + 3]) and (board[row][col] != 0):
                return board[row][col]  # Return Number that match row

    # Check columns for winner
    for col in range(7):
        for row in range(3):
            if (board[row][col] == board[row + 1][col] == board[row + 2][col] == \
                board[row + 3][col]) and (board[row][col] != 0):
                return board[row][col]  # Return Number that match column

    # Check diagonal (top-left to bottom-right) for winner

    for row in range(3):
        for col in range(4):
            if (board[row][col] == board[row + 1][col + 1] == board[row + 2][col + 2] == \
                board[row + 3][col + 3]) and (board[row][col] != 0):
                return board[row][col]  # Return Number that match diagonal

    # Check diagonal (bottom-left to top-right) for winner

    for row in range(5, 2, -1):
        for col in range(4):
            if (board[row][col] == board[row - 1][col + 1] == board[row - 2][col + 2] == \
                board[row - 3][col + 3]) and (board[row][col] != 0):
                return board[row][col]  # Return Number that match diagonal

    # No winner: return None
    return None
