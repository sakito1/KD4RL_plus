import pandas as pd
import os
import pickle as p
import numpy as np
from sklearn import preprocessing
from utils import str2num, num2str
from utils import cal

data_dir = './data/'

# tf.orthogonal_initializer

class Query(object):
    def __init__(self, train_start, split, input_len, hp, MC_thres_level, Credit_Option, use_feature_num):
        self.split_code = str2num(split)
        self.date_list = {}
        self.input_len = input_len
        self.hold_period = hp
        self.__features = None
        self.__MC_thres_level = MC_thres_level
        assert self.__MC_thres_level in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        assert Credit_Option in ['None', 'Credit', 'Exclude_Downgrades']
        # 0 represents none, 1 represents exclude ones without credit, 2 represents exclude downgrades

        MC_thres = pd.read_pickle(open(data_dir + 'MC_threshold-all.pkl', 'rb'))

        print('Credit Option is {}'.format(Credit_Option))
        if Credit_Option != 'None':
            Credit_dic = pd.read_pickle(open(data_dir + '{}.pkl'.format(Credit_Option), 'rb'))

        with open(data_dir + f'month_data_feature{use_feature_num}.pkl', 'rb') as f:
            self.data = pd.read_pickle(f)
            print('load initial file done')

            for k in list(self.data.keys()):
                # 1960 + [85/12] = 1967 , 85 % 12 = 1 , date >= 1967.1
                if k < str2num(train_start) + self.input_len or k + self.hold_period - 1 > str2num('2021-12'):
                    continue

                if self.__features is None:
                    self.__features = list(self.data[k].columns)
                else:
                    assert self.__features == list(self.data[k].columns)

                # exclude microcaps
                month_df = self.data[k]
                available_thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90]
                if self.__MC_thres_level == 0:
                    threshold = 0
                else:
                    assert self.__MC_thres_level in available_thresholds
                    threshold = MC_thres[k][available_thresholds.index(self.__MC_thres_level)]
                target_list = set(month_df[month_df.me >= threshold].index)
                # target_list = set(month_df.index)
                # print(len(target_list))

                for date_code in range(k - input_len, k + self.hold_period):
                    tmp_list = set(self.data[date_code].index)

                    target_list = target_list & tmp_list

                # print(len(target_list))

                # exclude credit samples
                if k >= self.split_code and Credit_Option != 'None':
                    target_list = set(Credit_dic[k]) & target_list

                self.date_list[k] = list(target_list)
                print('date : {} ,length is {}'.format(num2str(k), len(self.date_list[k])), end='\r')
            assert self.__features is not None and self.__features[0] == 'ret'
            print('Query init done')

    def concat2input(self, date_code, stock_list, scaler=preprocessing.StandardScaler):
        result = []
        my_scaler = scaler()
        for k in range(date_code - self.input_len, date_code):
            # print(num2str(k))
            df = self.data[k].loc[stock_list, self.__features]
            df_val = df.values

            df_val = my_scaler.fit_transform(df_val)

            result.append(df_val)

        df = self.data[date_code].loc[stock_list, self.__features]
        ret_vals = df.ret.values + 1
        for h in range(date_code + 1, date_code + self.hold_period):
            tmp_ret = self.data[h].loc[stock_list, ['ret']].ret.values + 1
            # print(np.mean(tmp_ret))
            ret_vals *= tmp_ret
            # print(np.mean(ret_vals))
        ret_vals -= 1
        df.ret = ret_vals

        result.append(df.values)

        # print(result[-1])
        # print(result[-2])
        result = np.array(result)
        # print(result.shape)
        result = np.rollaxis(result, 1)
        return result

    def one_step(self, begin, is_train=True):
        if begin not in self.date_list.keys():
            print(f'date must between {num2str(min(self.date_list.keys()))} and {num2str(max(self.date_list.keys()))}')
            print(num2str(begin))
            return None
        length = len(self.date_list[begin])
        if is_train:
            batch_index = [i for i in range(length)]
            np.random.shuffle(batch_index)
            fetch_list = [self.date_list[begin][i] for i in batch_index]
        else:
            fetch_list = list(self.date_list[begin])
        # np.random.shuffle(fetch_list)
        data = self.concat2input(begin, fetch_list)
        return data, fetch_list

    def next_batch_full(self, begin, time_step=12):
        batch = []
        slist = []
        for t in range(time_step):
            tmp, tl = self.one_step(begin + t, is_train=True)
            batch.append(tmp)
            slist.append(tl)
        return batch, slist

    def get_features(self):
        return self.__features

    def get_mes(self, date_code):
        mes = []
        for k in range(date_code, date_code + self.input_len):
            me_df = self.data[k - 1].loc[self.date_list[k], 'me']
            mes.append(me_df.values)
        return mes

    def get_month_me(self, date_code):
        return self.data[date_code - 1].loc[self.date_list[date_code], 'me'].values
