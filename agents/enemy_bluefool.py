"""
Source: https://www.kaggle.com/domcastro/connectx-rule-based-easy-solution-or-negamax
"""
from kaggle_environments.envs.connectx import connectx


def act(observation, configuration):
    # me:me_or_enemy=1, enemy:me_or_enemy=2
    def check_vertical_chance(me_or_enemy):
        for i in range(0, 7):
            if observation.board[i + 7 * 5] == me_or_enemy \
                    and observation.board[i + 7 * 4] == me_or_enemy \
                    and observation.board[i + 7 * 3] == me_or_enemy \
                    and observation.board[i + 7 * 2] == 0:
                return i
            elif observation.board[i + 7 * 4] == me_or_enemy \
                    and observation.board[i + 7 * 3] == me_or_enemy \
                    and observation.board[i + 7 * 2] == me_or_enemy \
                    and observation.board[i + 7 * 1] == 0:
                return i
            elif observation.board[i + 7 * 3] == me_or_enemy \
                    and observation.board[i + 7 * 2] == me_or_enemy \
                    and observation.board[i + 7 * 1] == me_or_enemy \
                    and observation.board[i + 7 * 0] == 0:
                return i
        # no chance
        return -99

    # me:me_or_enemy=1, enemy:me_or_enemy=2
    def check_horizontal_chance(me_or_enemy):
        chance_cell_num = -99
        for i in [0, 7, 14, 21, 28, 35]:
            for j in range(0, 4):
                val_1 = i + j + 0
                val_2 = i + j + 1
                val_3 = i + j + 2
                val_4 = i + j + 3
                if sum([observation.board[val_1] == me_or_enemy, \
                        observation.board[val_2] == me_or_enemy, \
                        observation.board[val_3] == me_or_enemy, \
                        observation.board[val_4] == me_or_enemy]) == 3:
                    for k in [val_1, val_2, val_3, val_4]:
                        if observation.board[k] == 0:
                            chance_cell_num = k
                            # bottom line
                            for l in range(35, 42):
                                if chance_cell_num == l:
                                    return l - 35
                            # others
                            if observation.board[chance_cell_num + 7] != 0:
                                return chance_cell_num % 7
        # no chance
        return -99

    # me:me_or_enemy=1, enemy:me_or_enemy=2
    def check_slanting_chance(me_or_enemy, lag, cell_list):
        chance_cell_num = -99
        for i in cell_list:
            val_1 = i + lag * 0
            val_2 = i + lag * 1
            val_3 = i + lag * 2
            val_4 = i + lag * 3
            if sum([observation.board[val_1] == me_or_enemy, \
                    observation.board[val_2] == me_or_enemy, \
                    observation.board[val_3] == me_or_enemy, \
                    observation.board[val_4] == me_or_enemy]) == 3:
                for j in [val_1, val_2, val_3, val_4]:
                    if observation.board[j] == 0:
                        chance_cell_num = j
                        # bottom line
                        for k in range(35, 42):
                            if chance_cell_num == k:
                                return k - 35
                        # others
                        if chance_cell_num != -99 \
                                and observation.board[chance_cell_num + 7] != 0:
                            return chance_cell_num % 7
        # no chance
        return -99

    def check_horizontal_first_enemy_chance():
        # enemy's chance
        if observation.board[38] == enemy_num:
            if sum([observation.board[39] == enemy_num, observation.board[40] == enemy_num]) == 1 \
                    and observation.board[37] == 0:
                for i in range(39, 41):
                    if observation.board[i] == 0:
                        return i - 35
            if sum([observation.board[36] == enemy_num, observation.board[37] == enemy_num]) == 1 \
                    and observation.board[39] == 0:
                for i in range(36, 38):
                    if observation.board[i] == 0:
                        return i - 35
        # no chance
        return -99

    def check_first_or_second():
        count = 0
        for i in observation.board:
            if i != 0:
                count += 1
        # first
        if count % 2 != 1:
            my_num = 1
            enemy_num = 2
        # second
        else:
            my_num = 2
            enemy_num = 1
        return my_num, enemy_num

    # check first or second
    my_num, enemy_num = check_first_or_second()

    def check_my_chances():
        # check my virtical chance
        result = check_vertical_chance(my_num)
        if result != -99:
            return result
        # check my horizontal chance
        result = check_horizontal_chance(my_num)
        if result != -99:
            return result
        # check my slanting chance 1 (up-right to down-left)
        result = check_slanting_chance(my_num, 6, [3, 4, 5, 6, 10, 11, 12, 13, 17, 18, 19, 20])
        if result != -99:
            return result
        # check my slanting chance 2 (up-left to down-right)
        result = check_slanting_chance(my_num, 8, [0, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 17])
        if result != -99:
            return result
        # no chance
        return -99

    def check_enemy_chances():
        # check horizontal first chance
        result = check_horizontal_first_enemy_chance()
        if result != -99:
            return result
        # check enemy's vertical chance
        result = check_vertical_chance(enemy_num)
        if result != -99:
            return result
        # check enemy's horizontal chance
        result = check_horizontal_chance(enemy_num)
        if result != -99:
            return result
        # check enemy's slanting chance 1 (up-right to down-left)
        result = check_slanting_chance(enemy_num, 6, [3, 4, 5, 6, 10, 11, 12, 13, 17, 18, 19, 20])
        if result != -99:
            return result
        # check enemy's slanting chance 2 (up-left to down-right)
        result = check_slanting_chance(enemy_num, 8, [0, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 17])
        if result != -99:
            return result
        # no chance
        return -99

    if my_num == 1:
        result = check_my_chances()
        if result != -99:
            return result
        result = check_enemy_chances()
        if result != -99:
            return result
    if my_num == 2:
        result = check_enemy_chances()
        if result != -99:
            return result
        result = check_my_chances()
        if result != -99:
            return result

    # select center as priority (3 > 2 > 4 > 1 > 5 > 0 > 6)
    # column 3
    if observation.board[24] != enemy_num \
            and observation.board[17] != enemy_num \
            and observation.board[10] != enemy_num \
            and observation.board[3] == 0:
        return 3
    # column 2
    elif observation.board[23] != enemy_num \
            and observation.board[16] != enemy_num \
            and observation.board[9] != enemy_num \
            and observation.board[2] == 0:
        return 2
    # column 4
    elif observation.board[25] != enemy_num \
            and observation.board[18] != enemy_num \
            and observation.board[11] != enemy_num \
            and observation.board[4] == 0:
        return 4
    # column 1
    elif observation.board[22] != enemy_num \
            and observation.board[15] != enemy_num \
            and observation.board[8] != enemy_num \
            and observation.board[1] == 0:
        return 1
    # column 5
    elif observation.board[26] != enemy_num \
            and observation.board[19] != enemy_num \
            and observation.board[12] != enemy_num \
            and observation.board[5] == 0:
        return 5
    # column 0
    elif observation.board[21] != enemy_num \
            and observation.board[14] != enemy_num \
            and observation.board[7] != enemy_num \
            and observation.board[0] == 0:
        return 0
    # column 6
    elif observation.board[27] != enemy_num \
            and observation.board[20] != enemy_num \
            and observation.board[13] != enemy_num \
            and observation.board[6] == 0:
        return 6
    # random
    else:
        return connectx.negamax_agent(observation, configuration)
