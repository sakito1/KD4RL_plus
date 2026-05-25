import argparse
import json
import os
import copy
import time
from datetime import datetime
import logging
import configparser
from tqdm import *
import pandas as pd
import numpy as np
# from torch.utils.tensorboard import SummaryWriter

from utils.parse_config import ConfigParser
from utils.functions import *
from agent import *
from environment.portfolio_env import PortfolioEnv



def run(func_args):
    if func_args.seed != -1:
        setup_seed(func_args.seed)

    data_prefix = './data/' + func_args.market + '/'
    matrix_path = data_prefix + func_args.relation_file

    start_time = datetime.now().strftime('%m%d/%H:%M:%S')
    if func_args.mode == 'train':
        output_root = getattr(func_args, 'output_dir', None) or 'outputs/'
        PREFIX = os.path.join(output_root, start_time)
        img_dir = os.path.join(PREFIX, 'img_file')
        save_dir = os.path.join(PREFIX, 'log_file')
        model_save_dir = os.path.join(PREFIX, 'model_file')

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)
        if not os.path.isdir(model_save_dir):
            os.mkdir(model_save_dir)

        hyper = copy.deepcopy(func_args.__dict__)
        print(hyper)
        hyper['device'] = 'cuda' if hyper['device'] == torch.device('cuda') else 'cpu'
        json_str = json.dumps(hyper, indent=4)

        with open(os.path.join(save_dir, 'hyper.json'), 'w') as json_file:
            json_file.write(json_str)

        # writer = SummaryWriter(save_dir)
        # writer.add_text('hyper_setting', str(hyper))

        logger = logging.getLogger()
        logger.setLevel('INFO')
        BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
        DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
        chlr = logging.StreamHandler()
        chlr.setFormatter(formatter)
        chlr.setLevel('WARNING')
        fhlr = logging.FileHandler(os.path.join(save_dir, 'logger.log'))
        fhlr.setFormatter(formatter)
        logger.addHandler(chlr)
        logger.addHandler(fhlr)



        stocks_data = np.load(data_prefix + 'features.npy')
        rate_of_return = np.load(data_prefix + 'rets.npy')
        market_history = None
        assert stocks_data.shape[:-1] == rate_of_return.shape, 'file size error'
        A = torch.from_numpy(np.load(matrix_path)).float().to(func_args.device)
        split_config = configparser.ConfigParser()
        split_path = os.path.join(data_prefix, 'split_idx.txt')
        split_config.read(split_path)
        val_idx = split_config.getint('valid', 'start')
        test_idx = split_config.getint('test', 'start')
        test_end_idx = split_config.getint('test', 'end_excl')
        allow_short = False


        env = PortfolioEnv(assets_data=stocks_data, market_data=market_history, rtns_data=rate_of_return,
                           in_features=func_args.in_features, val_idx=val_idx, test_idx=test_idx,test_end_idx = test_end_idx,
                           batch_size=func_args.batch_size, window_len=func_args.window_len, trade_len=func_args.trade_len,
                           max_steps=func_args.max_steps, mode=func_args.mode, norm_type=func_args.norm_type,
                           allow_short=allow_short)

        supports = [A]
        actor = RLActor(supports, func_args).to(func_args.device)
        agent = RLAgent(env, actor, func_args)

        mini_batch_num = int(np.ceil(len(env.src.order_set) / func_args.batch_size))
        max_batches = getattr(func_args, 'max_batches', None)
        if max_batches is not None:
            mini_batch_num = min(mini_batch_num, int(max_batches))
        try:
            max_cr = 0
            metric_rows = []
            for epoch in range(func_args.epochs):
                epoch_return = 0
                for j in tqdm(range(mini_batch_num)):
                    episode_return, avg_rho, avg_mdd = agent.train_episode()
                    epoch_return += episode_return
                avg_train_return = epoch_return / mini_batch_num
                logger.warning('[%s]round %d, avg train return %.4f, avg rho %.4f, avg mdd %.4f' %
                               (start_time, epoch, avg_train_return, avg_rho, avg_mdd))



                test_wealth, val_wealth, test_wealth_list, val_wealth_list = agent.evaluation()

                # # 保存验证集/测试集的每日组合收益率（用于日度年化指标）
                # if getattr(agent, 'val_daily_ret', None) is not None:
                #     np.save(os.path.join(save_dir, f'val_daily_ret_epoch{epoch}.npy'), agent.val_daily_ret)
                # if getattr(agent, 'test_daily_ret', None) is not None:
                #     np.save(os.path.join(save_dir, f'test_daily_ret_epoch{epoch}.npy'), agent.test_daily_ret)
                val_metrics = {
                    key: float(np.asarray(value).squeeze())
                    for key, value in calculate_daliy_metrics(val_wealth_list, func_args.trade_mode).items()
                }
                test_metrics = {
                    key: float(np.asarray(value).squeeze())
                    for key, value in calculate_daliy_metrics(test_wealth_list, func_args.trade_mode).items()
                }
                metric_rows.append({
                    'epoch': epoch,
                    **{f'val_{key}': value for key, value in val_metrics.items()},
                    **{f'test_{key}': value for key, value in test_metrics.items()},
                })

                print('Val/APR', val_metrics['APR'])
                print('Val/MDD', val_metrics['MDD'])
                print('Val/AVOL', val_metrics['AVOL'])
                print('Val/ASR', val_metrics['ASR'])
                print('Val/SoR', val_metrics['DDR'])
                print('Val/CR', val_metrics['CR'])

                print('Test/APR', test_metrics['APR'])
                print('Test/MDD', test_metrics['MDD'])
                print('Test/AVOL', test_metrics['AVOL'])
                print('Test/ASR', test_metrics['ASR'])
                print('Test/SoR', test_metrics['DDR'])
                print('Test/CR', test_metrics['CR'])

                # 用 Val 的 CR 做 best model 选择（推荐）
                if val_metrics['CR'] > max_cr:
                    print('New Best (Val CR) Policy!!!!')
                    max_cr = val_metrics['CR']
                    torch.save(actor, os.path.join(model_save_dir, 'best_val_cr-' + str(epoch) + '.pkl'))

                # logger：建议同时打 val/test；你如果只想打一个，就选 val（训练过程中更合理）
                logger.warning(
                    'after training %d round, '
                    '[VAL] max wealth: %.4f, min wealth: %.4f, avg wealth: %.4f, final wealth: %.4f, '
                    'ARR: %.3f%%, ASR: %.3f, AVol: %.3f, MDD: %.2f%%, CR: %.3f, DDR: %.3f | '
                    '[TEST] max wealth: %.4f, min wealth: %.4f, avg wealth: %.4f, final wealth: %.4f, '
                    'ARR: %.3f%%, ASR: %.3f, AVol: %.3f, MDD: %.2f%%, CR: %.3f, DDR: %.3f'
                    % (
                        epoch,
                        max(val_wealth[0]), min(val_wealth[0]), np.mean(val_wealth), val_wealth[-1, -1],
                        100 * val_metrics['APR'], val_metrics['ASR'], val_metrics['AVOL'],
                        100 * val_metrics['MDD'], val_metrics['CR'], val_metrics['DDR'],

                        max(test_wealth[0]), min(test_wealth[0]), np.mean(test_wealth), test_wealth[-1, -1],
                        100 * test_metrics['APR'], test_metrics['ASR'], test_metrics['AVOL'],
                        100 * test_metrics['MDD'], test_metrics['CR'], test_metrics['DDR'],
                    )
                )
            pd.DataFrame(metric_rows).to_csv(os.path.join(save_dir, 'performance.csv'), index=False)
            torch.save(actor, os.path.join(model_save_dir, 'final_model.pkl'))
            torch.save(agent.optimizer.state_dict(), os.path.join(model_save_dir, 'final_optimizer.pkl'))
        except KeyboardInterrupt:
            torch.save(actor, os.path.join(model_save_dir, 'final_model.pkl'))
            torch.save(agent.optimizer.state_dict(), os.path.join(model_save_dir, 'final_optimizer.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('--window_len', type=int)
    parser.add_argument('--G', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--no_spatial', dest='spatial_bool', action='store_false')
    parser.add_argument('--no_msu', dest='msu_bool', action='store_true')
    parser.add_argument('--relation_file', type=str)
    parser.add_argument('--addaptiveadj', dest='addaptive_adj_bool', action='store_false')
    parser.add_argument('--output_dir', type=str)

    opts = parser.parse_args()

    if opts.config is not None:
        with open(opts.config) as f:
            options = json.load(f)
            args = ConfigParser(options)
    else:
        with open('./hyper.json') as f:
            options = json.load(f)
            args = ConfigParser(options)
    args.update(opts)

    run(args)
