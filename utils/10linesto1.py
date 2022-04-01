# open file  kp20k
import numpy as np
with open('./concatenated.txt', 'w', newline='') as output:
    with open('./kp20k_valid_m8_step_100000') as f:
        lines = f.readlines()
        output_line = []

        for i in range(len(lines) // 10):
            lignes = lines[i:i+10]
            lignes10 = np.concatenate(lignes)
            output_line.append(np.r)
            # remove duplicate
