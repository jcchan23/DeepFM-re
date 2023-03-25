#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   split_dataset.py
@Time    :   2023/03/17 14:27:31
@Author  :   Jianwen Chen
@Version :   1.0
@Contact :   chenjw48@mail3.sysu.edu.cn
@License :   (C)Copyright 2023, 厚朴【HOPE】工作室, SAIL-LAB
@Desc    :   None
'''

######################################## import area ########################################

import sys
import subprocess
import random
from tqdm import tqdm

######################################## parser area ########################################


######################################## function area ########################################


######################################## main area ########################################

if __name__ == '__main__':
    
    path = './data/source'
    test_size = 0.1
    
    print('split dataset begin!')
    total_cnt = int(subprocess.getoutput(f'wc -l {path}/{sys.argv[1]}/{sys.argv[1]}.txt').split()[0])
    with open(f"{path}/{sys.argv[1]}/{sys.argv[1]}.txt", 'r') as f, \
        open(f"{path}/{sys.argv[1]}/{sys.argv[1]}_train.txt", 'w') as train_fw, \
        open(f"{path}/{sys.argv[1]}/{sys.argv[1]}_test.txt", 'w') as test_fw:
        
        for idx, line in tqdm(enumerate(f.readlines()), total=total_cnt):
            r = random.random()
            if r <= test_size:
                test_fw.write(line)
            else:
                train_fw.write(line)
            if idx > 0 and idx % 1e+7 == 0:
                print(f'split dataset {idx} rows!')
    
    print('split dataset finish!')
