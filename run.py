#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   run.py
@Time    :   2023/03/21 14:18:23
@Author  :   Jianwen Chen
@Version :   1.0
@Contact :   chenjw48@mail3.sysu.edu.cn
@License :   (C)Copyright 2023, 厚朴【HOPE】工作室, SAIL-LAB
@Desc    :   None
'''

######################################## import area ########################################

import os
import sys
import yaml
import pickle
import random
import subprocess
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tqdm import tqdm
from collections import Counter
from tensorflow import feature_column
from tensorflow.keras.layers import Input, DenseFeatures, Dense, Concatenate, Flatten, Add, Subtract, Multiply, Lambda, Dropout, Activation
from tensorflow.keras.models import Model

######################################## parser area ########################################

with open(sys.argv[1], 'r', encoding='UTF-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

######################################## function area ########################################

def build_data(src_data_path, dst_data_path, mode, sep=',', chunksize=config["chunk_size"]):

    statis_info = dict()

    if mode == 'test':

        with open(dst_data_path, 'w') as fw:
            total_cnt = int(subprocess.getoutput(f"wc -l {src_data_path}").split()[0])
            reader = pd.read_csv(src_data_path, sep=sep, names=['label'] + config["continuous_features"] + config["categorial_features"], chunksize=chunksize)
            print(f"There will be {total_cnt // chunksize + 1} data blocks!")
            for data in tqdm(reader):

                data[config["continuous_features"]] = data[config["continuous_features"]].fillna(0)
                data[config["categorial_features"]] = data[config["categorial_features"]].fillna('<unk>')

                for row in data.itertuples():
                    fw.write(",".join([str(num) for num in list(row)[1:]]) + '\n')
        
    else:

        for column in config["continuous_features"] + config["categorial_features"]:
            if column in config["continuous_features"]:
                statis_info[column] = {'min':float('inf'), 'max':float('-inf')}
            else:
                statis_info[column] = dict()
        
        with open(dst_data_path, 'w') as fw:
            total_cnt = int(subprocess.getoutput(f"wc -l {src_data_path}").split()[0])
            reader = pd.read_csv(src_data_path, sep=sep, names=['label'] + config["continuous_features"] + config["categorial_features"], chunksize=chunksize)
            print(f"There will be {total_cnt // chunksize + 1} data blocks!")
            for data in tqdm(reader):

                data[config["continuous_features"]] = data[config["continuous_features"]].fillna(0)
                data[config["categorial_features"]] = data[config["categorial_features"]].fillna('<unk>')

                for column in config["continuous_features"]:
                    data[column] = data[column].apply(lambda x: x if x <= config["continuous_clip"][column] else config["continuous_clip"][column])
                    statis_info[column]['min'] = min(statis_info[column]['min'], min(data[column].values))
                    statis_info[column]['max'] = max(statis_info[column]['max'], max(data[column].values))

                for column in config["categorial_features"]:
                    for feature, value in Counter(list(data[column].values)).items():
                        if feature not in statis_info[column].keys():
                            statis_info[column][feature] = value
                        else:
                            statis_info[column][feature] += value

                for row in data.itertuples():
                    fw.write(",".join([str(num) for num in list(row)[1:]]) + '\n')

    return statis_info
        

def build_features(statis_info):

    features_columns = dict()
    features_layer_inputs = dict()

    for column in config["continuous_features"]:
        
        features_layer_inputs[column] = Input(shape=1, dtype=tf.float32, name=column)

        min_num, max_num = statis_info[column]['min'], statis_info[column]['max']

        features_columns[column] = feature_column.numeric_column(column, default_value=0.0, normalizer_fn = lambda x: (x - min_num)/(max_num - min_num))

    for column in config["categorial_features"]:

        features_layer_inputs[column] = Input(shape=1, dtype=tf.string, name=column)

        cnt_dict = filter(lambda x: x[1] >= config["categorial_clip"], statis_info[column].items())
        cnt_dict = dict(sorted(cnt_dict, key=lambda x: (-x[1], x[0])))
        
        categorial_column = feature_column.categorical_column_with_vocabulary_list(column, vocabulary_list = list(cnt_dict.keys()), default_value = 0)
        features_columns[column] = feature_column.embedding_column(categorial_column, dimension=config["1d_embedding_dim"], trainable=True)
    
    return features_columns, features_layer_inputs


def data_generator(file_path, batch_size=config["batch_size"]):
    
    while True:
        reader = pd.read_csv(file_path, sep=',', names=['label'] + config["continuous_features"] + config["categorial_features"], chunksize=batch_size)
        for chunk in reader:
            continuous_dict = {column: tf.constant(chunk[column].values, dtype=tf.float32) for column in config["continuous_features"]}
            categorial_dict = {column: tf.constant(chunk[column].values, dtype=tf.string) for column in config["categorial_features"]}
            label_dict = {'output': tf.constant(chunk['label'].values, dtype=tf.int32)}
            yield ({**continuous_dict, **categorial_dict}, label_dict)


######################################## main area ########################################

if __name__ == "__main__":
    
    # build statistics infomation for feature preprocess
    if not os.path.exists(f'{config["dst_path"]}/{config["dataset"]}/{config["dataset"]}_statis_info.pickle') or config['rebuild_statis_info']:
        statis_info = build_data(f'{config["src_path"]}/{config["dataset"]}/{config["dataset"]}_train.txt', 
                                f'{config["dst_path"]}/{config["dataset"]}/{config["dataset"]}_train_valid.txt', mode='train', sep='\t', chunksize=config["chunk_size"])
        
        with open(f'{config["dst_path"]}/{config["dataset"]}/{config["dataset"]}_statis_info.pickle', 'wb') as fw:
            pickle.dump(statis_info, fw)
    else:
        with open(f'{config["dst_path"]}/{config["dataset"]}/{config["dataset"]}_statis_info.pickle', 'rb') as f:
            statis_info = pickle.load(f)
        
    # build schema
    feature_columns, features_layer_inputs = build_features(statis_info)

    '''
        Embedding层
    '''
    dense_output = [DenseFeatures(feature_columns[column])({column: features_layer_inputs[column]}) for column in config["continuous_features"]]
    sparse_1d_output = [DenseFeatures(feature_columns[column])({column: features_layer_inputs[column]}) for column in config["categorial_features"]]

    '''
        一阶特征交叉
    '''
    # list([batch, 1]) -> [batch, 13]
    first_order_dense_layer = Dropout(config["embedding_dropout_rate"])(Concatenate(axis=1)(dense_output))

    # list([batch, 1d_embedding_dim]) -> [batch, 26 * 1d_embedding_dim]
    first_order_sparse_layer = Dropout(config["embedding_dropout_rate"])(Concatenate(axis=1)(sparse_1d_output))

    # [batch, 13 + 26 * 1d_embedding_dim] -> [batch, 1]
    first_order_layer = Concatenate(axis=1)([first_order_dense_layer, first_order_sparse_layer])
    first_order_layer = Dense(1, kernel_regularizer=tf.keras.regularizers.l2(config["l2_regular_rate"]))(first_order_layer)

    '''
        二阶特征交叉，仅作用于类别特征。
    '''
    # list([batch, kd_embedding_dim]) -> list([batch, 1, kd_embedding_dim]) -> [batch, 26, kd_embedding_dim]
    sparse_kd_output = [Dense(config["kd_embedding_dim"], kernel_regularizer=tf.keras.regularizers.l2(config["l2_regular_rate"]))(e) for e in sparse_1d_output]
    concat_sparse_kd_embeds = Dropout(config["embedding_dropout_rate"])(Concatenate(axis=1)([e[:, tf.newaxis, :] for e in sparse_kd_output]))

    # [batch, kd_embedding_dim]，先求和再平方
    sum_sparse_kd_embeds = Lambda(lambda x: K.sum(x, axis=1))(concat_sparse_kd_embeds)
    square_sum_sparse_kd_embeds = Multiply()([sum_sparse_kd_embeds, sum_sparse_kd_embeds])

    # [batch, kd_embedding_dim]，先平方再求和
    square_sparse_kd_embeds = Multiply()([concat_sparse_kd_embeds, concat_sparse_kd_embeds])
    sum_square_sparse_kd_embeds = Lambda(lambda x: K.sum(x, axis=1))(square_sparse_kd_embeds)

    # [batch, kd_embedding_dim]，相减除以2
    sub_sparse_kd_embeds = Subtract()([square_sum_sparse_kd_embeds, sum_square_sparse_kd_embeds])
    sub_sparse_kd_embeds = Lambda(lambda x: x * 0.5)(sub_sparse_kd_embeds)

    # [batch, kd_embedding_dim] -> [batch, 1]
    second_order_sparse_layer = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(sub_sparse_kd_embeds)

    '''
        DNN特征，需要将一阶连续特征和一阶类别特征拼接起来输入到dnn中。
        实验证明这里使用经过二阶特征的输入比一阶特征的输入要好。
    '''
    # [batch, 13 + 26 * kd_embedding_dim]
    concat_sparse_dnn_embeds = Flatten()(concat_sparse_kd_embeds)
    dnn_layer = Concatenate(axis=1)([first_order_dense_layer, concat_sparse_dnn_embeds])
    dnn_layer = Dropout(config["dnn_dropout_rate"])(Dense(config["dnn_dim"], activation='relu')(dnn_layer))
    dnn_layer = Dropout(config["dnn_dropout_rate"])(Dense(config["dnn_dim"], activation='relu')(dnn_layer))
    dnn_layer = Dense(1, kernel_regularizer=tf.keras.regularizers.l2(config["l2_regular_rate"]))(dnn_layer)

    '''
        合并fm层的输出和dnn层的输出
    '''
    output = Add()([first_order_layer, second_order_sparse_layer, dnn_layer])
    # 这里的name需要和generator的输出一致，否则generator无法给出对应的数据。
    output = Activation('sigmoid', name='output')(output)

    # 初始化模型
    model = Model(inputs = features_layer_inputs, outputs = output)
    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(),
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
        metrics = [
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.BinaryCrossentropy()
        ]
    )
    model.summary()

    # EarlyStop配置
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, mode='max', min_lr=1e-4)

    # 切分文件
    with open(f'{config["dst_path"]}/{config["dataset"]}/{config["dataset"]}_train_valid.txt', 'r') as f, \
        open(f'{config["dst_path"]}/{config["dataset"]}/{config["dataset"]}_train.txt', 'w') as train_fw, \
        open(f'{config["dst_path"]}/{config["dataset"]}/{config["dataset"]}_valid.txt', 'w') as valid_fw:

        for idx, line in enumerate(f.readlines()):
            r = random.random()
            if r <= 0.1:
                valid_fw.write(line)
            else:
                train_fw.write(line)
            if idx > 0 and idx % 1e+7 == 0:
                print(f'split dataset {idx} rows!')
    
    train_count = int(subprocess.getoutput(f'wc -l {config["dst_path"]}/{config["dataset"]}/{config["dataset"]}_train.txt').split()[0])
    valid_count = int(subprocess.getoutput(f'wc -l {config["dst_path"]}/{config["dataset"]}/{config["dataset"]}_valid.txt').split()[0])
    print(f'train size = {train_count} | valid size = {valid_count}')

    # 训练与评估
    model.fit(
        data_generator(f'{config["dst_path"]}/{config["dataset"]}/{config["dataset"]}_train.txt', batch_size=config["batch_size"]),
        steps_per_epoch = train_count // config["batch_size"],
        validation_data = data_generator(f'{config["dst_path"]}/{config["dataset"]}/{config["dataset"]}_valid.txt', batch_size=config["batch_size"]),
        validation_steps = valid_count // config["batch_size"],
        epochs=config["epochs"], 
        verbose=2, 
        callbacks=[early_stop, reduce_lr]
    )

    # 测试
    build_data(f'{config["src_path"]}/{config["dataset"]}/{config["dataset"]}_test.txt', f'{config["dst_path"]}/{config["dataset"]}/{config["dataset"]}_test.txt', mode='test', sep='\t', chunksize=config["chunk_size"])
    test_count = int(subprocess.getoutput(f'wc -l {config["dst_path"]}/{config["dataset"]}/{config["dataset"]}_test.txt').split()[0])
    print(f'test size = {test_count}')
    model.evaluate(
        data_generator(f'{config["dst_path"]}/{config["dataset"]}/{config["dataset"]}_test.txt', batch_size=config["batch_size"]), 
        steps = test_count // config["batch_size"], 
        verbose=2
    )
