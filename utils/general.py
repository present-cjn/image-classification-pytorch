# -*- coding: utf-8 -*-
"""
Time:     2023/11/30 14:42
Author:   cjn
Version:  1.0.0
File:     general.py
Describe: 
"""
import os

def create_experiment_folder(base_dir):
    # 获取`runs`目录下以`experiment_`为前缀的文件夹列表
    experiment_folders = [f for f in os.listdir(base_dir) if f.startswith('experiment_')]

    if len(experiment_folders) == 0:
        # 如果没有以`experiment_`为前缀的文件夹，则创建`experiment_1`
        new_folder_name = 'experiment_1'
    else:
        # 获取已存在的最大序号
        existing_numbers = [int(f.split('_')[1]) for f in experiment_folders]
        max_number = max(existing_numbers)

        # 创建新的文件夹序号
        new_number = max_number + 1
        new_folder_name = f'experiment_{new_number}'

    # 在`runs`目录下创建新的文件夹
    new_folder_path = os.path.join(base_dir, new_folder_name)
    os.makedirs(new_folder_path)

    return new_folder_path