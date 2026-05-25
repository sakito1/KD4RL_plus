import tensorflow as tf
import os
import argparse
from network import lstm_drl
from input import Query
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
import pandas as pd
from copy import deepcopy
from utils import *
import heapq
import pickle

val_dic = {0:'ret',1:'std',2:'vol',3:'me',4:'B2M',5:'PE', 6:'Div'}

parser = argparse.ArgumentParser(description='DRL_finance TensorFlow implementation.')

parser.add_argument('--mode',               type=str,   help='train or test', default='train')
parser.add_argument('--model_name',         type=str,   help='model name', default='LSTM-DRL')
# parser.add_argument('--batch_size',         type=int,   help='batch size', default=1000)
parser.add_argument('--learning_rate',      type=float,help='init learning rate',default=1e-4)
parser.add_argument('--hidden_size1',       type=int,   help='number of hidden units of lstm layer 1',default=64)
parser.add_argument('--hidden_size2',       type=int,   help='number of hidden units of lstm layer 2',default=128)
parser.add_argument('--trade_por' ,         type=float, help='trade_num / batch_size',default=0.25)
parser.add_argument('--lookup_size' ,        type=int, help='size of the lookup table',default=50)
parser.add_argument('--input_len',          type=int,   help='input time step length',default=12)
parser.add_argument('--trade_len',          type=int,   help='trade time step length',default=12)
parser.add_argument('--num_epochs',         type=int,   help='number of epochs', default=30)
parser.add_argument('--gpu',                type=str,   help='number of GPUs to use for training', default='0')
parser.add_argument('--log_directory',      type=str,   help='directory to save checkpoints and summaries', default='/data/zhangyang/DRL_models/2-1/tl_3/model_saved/')
parser.add_argument('--checkpoint_path',    type=str,   help='path to a specific checkpoint to load', default='model_saved/')
parser.add_argument('--retrain',            help='if used with checkpoint_path, will restart training from step zero', action='store_true')

args = parser.parse_args()
query = Query(split='1990-1',input_len=args.input_len)
# query = Query(input_len=args.input_len)

keep_prob = tf.placeholder(tf.float32)

params = {
        'hidden_size1': args.hidden_size1,
        'hidden_size2': args.hidden_size2,
        'lookup_size' : args.lookup_size
    }

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def train():
    print('----------training----------')

    inputs = [tf.placeholder(tf.float32, [None, args.input_len + 1, 7]) for _ in range(args.trade_len)]
    trade_num = [tf.placeholder(tf.int32) for _ in range(args.trade_len)]

    model = lstm_drl(inputs=inputs, keep_prob=keep_prob, params=params, trade_num=trade_num)

    global_step = tf.Variable(0, trainable=False)
    lr = args.learning_rate
    learning_rate = tf.train.piecewise_constant(global_step, boundaries=[1000, 3000],
                                                values=[lr, lr*0.5 , lr*0.1])

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(model.loss, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=15)
    if not os.path.exists(args.log_directory):
        os.mkdir(args.log_directory)
    save_path = args.log_directory + 'model_fin.ckpt'
    print('save path is {}'.format(save_path))

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess,save_path='./model_saved2/model_fin.ckpt-1590')
        date_range = [i for i in range(query.split_code - 12*19 , query.split_code-args.trade_len+1)]
        print('training period is between {} and {} ,length is {}'.format(num2str(date_range[0]),num2str(date_range[-1]),len(date_range)))
        for epoch in range(args.num_epochs):
            np.random.shuffle(date_range)
            epoch_loss = 0.0
            for d in date_range:

                batch_data,sr_data ,_ = query.next_batch_full(d, time_step=args.trade_len)
                tr_num = [int(len(month_data)*args.trade_por) for month_data in batch_data]
                # print(prc_rank[0])

                fd = {p:v for p,v in zip(inputs,batch_data)}
                fd.update({tn:v for tn,v in zip(trade_num,tr_num)})
                fd.update({keep_prob:0.5})
                l,s, _ = sess.run([model.loss,global_step,train_op],feed_dict=fd)
                epoch_loss += l
                if s % 10 == 0:
                    print('epoch {} - step {} , loss is {}'.format(epoch,s,l))

            print('epoch is {} , loss is {}'.format(epoch,epoch_loss/len(date_range)))
            if (epoch >0 and epoch %5==0) or (epoch_loss/len(date_range)<-4):
                s = sess.run(global_step)
                saver.save(sess,save_path,global_step=s)
                print('epoch {} , save model done'.format(epoch))

        s = sess.run(global_step)
        saver.save(sess,save_path,global_step=s)
        print('training done , model saved')




def eval():
    print('----------testing-----------')

    inputs = [tf.placeholder(tf.float32, [None, args.input_len + 1, 7])]
    trade_num = [tf.placeholder(tf.int32)]

    model = lstm_drl(inputs=inputs, keep_prob=keep_prob, params=params, trade_num=trade_num)

    latest_ckpt = tf.train.latest_checkpoint('/data/zhangyang/DRL_models/2-1/tl_15/'+args.checkpoint_path)
    # latest_ckpt = '/data/zhangyang/DRL_models/2-1/tl_18/' + args.checkpoint_path + 'model_fin.ckpt-{}'.format(2321)

    saver = tf.train.Saver()

    date_list = deepcopy(query.date_list)
    with tf.Session(config=config) as sess:
        saver.restore(sess,latest_ckpt)
        print('load done , model path is {}'.format(latest_ckpt))
        date_range = [i for i in list(date_list.keys()) if i >= query.split_code
                        and i <query.split_code+12*30]
        print('test period is between {} and {}'.format(num2str(date_range[0]),num2str(date_range[-1])))
        ret = []
        # regress_data = []
        grads = None
        for d in date_range:
            bat_size = len(query.date_list[d])
            tr_num = [int(bat_size * args.trade_por)]
            batch_data , sr_data, _ = query.one_step(d,False,bat_size)


            fd = {p: v for p, v in zip(inputs, [batch_data])}
            fd.update({tn: v for tn, v in zip(trade_num, tr_num)})

            fd.update({keep_prob: 1.0})

            month_ret , stock_score , attn,ws  = sess.run([model.portfolio_return , model.stock_score,model.attention_weight ,model.pred ] , feed_dict=fd)

            # np.save('/data/zhangyang/results/attn.npy' ,attn)
            # np.save('/data/zhangyang/results/ws.npy',ws)
            # print('save done')
            # os._exit(0)

            grad = np.reshape(sess.run(model.gradients,feed_dict=fd),[-1,args.input_len,7])
            # print(grad[0,:,1])
            # os._exit(0)

            if grads is None:
                grads = grad
            else:
                grads = np.concatenate([grads,grad],axis=0)

            assert len(month_ret) == 1
            ret.extend(month_ret)

            # tmp_list = []
            # for b in range(stock_score.shape[1]):
            #     tmp_dict = {}
            #     tmp_dict['score'] = stock_score[0,b]
            #     for t in range(args.input_len):
            #         for v in range(7):
            #             tmp_dict[val_dic[v]+ '_' +str(t)] = batch_data[b,t,v]
            #
            #     tmp_list.append(tmp_dict)

            # regress_data.extend(tmp_list)

        print(grads.shape)
        print(np.mean(grads[:,:,1],axis=0))
        # np.save('./dydx.npy',grads)
        # regress = pd.DataFrame(regress_data)
        # regress.to_csv('./regress.csv',index=None)
        print('length of ret is {}'.format(len(ret)))
        ret = [i/2 for i in ret]
        # print(ret)
        # np.save('./DRL_FDDR.npy',np.array(ret))
        # print('save done')
        ret_cum = np.cumsum(np.array([0] + ret)) + 1
        ans = cal(ret , ret_cum)
        print('evaluation result is : ')

        print(ans)

        # begin,end = 2010,2015
        # np.save('./results/{}-{}.npy'.format(begin,end),np.array(ret))
        # print('save result done of {}-{}'.format(begin,end))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        eval()
