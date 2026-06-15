# base parameters (do not modify)
train_start_date = "2000-04-07"
train_end_date = "2017-12-29"
# 验证集包含Nas100牛市两年(nasdaq100在2009到2021年整体上涨，在2020年4月22日前后有较大下跌)
valid_start_date = "2018-01-02"
valid_end_date = "2020-04-22"
test_start_date = "2020-04-23"
test_end_date = "2025-10-03"

seed = 42

if_use_per = False
if_norm = True
if_norm_temporal = False
red=0.0 #用来表示标注的阈值

# deeptrader data path
deeptrader_data_path = "DeepTrader/src/data/NAS/"

# ssm model parameters
ssm_encoder_window = 21 # encoder编码的窗口大小

# train parameters (adjust mainly)
days = 720 # 用于平滑每天的reward，用来防止reward波动过大
inner_batch_size = 1000 # 每次抽取的batch大小
outer_batch_size = 480
train_episodes = 5 # 训练的episode
inner_train_episodes = 5 # 训练的episode
test_episodes = 1 # 测试的episode
embed_dim = 32 #
lr = 1e-3 # 特征提取网络学习率
lr_actor = 3e-4 # actor学习率
lr_critic = 3e-4 # critic学习率
lr_alpha = 3e-4 # alpha学习率
print_interval = 10

# 关于交易的参数
min_trade_days = 720 # 最小交易天数
initial_amount = 1e3 # 初始资金
window_size=30
inner_CAPACITY=2000 # 经验回放池的大小
outer_CAPACITY=1000 # 经验回放池的大小
gamma=0.5 # 折扣因子
gamma1=0.99
gamma2=0.5
tau=0.01 # 软更新参数
target_entropy= -5
num_nodes = 39 #股票池大小
inner_max_boundary = 0.6 # 内层每次交易的最大比例
# TODO:分层强化学习参数, 先固定窗口大小
outer_window = 240
inner_window = 10
outer_horizon = 60
max_hold = 40 # 最大持有期
min_hold = 10
warmup_monitor_force_hold_prob = 0.0
episode_len=960
# 通过env的observation获取的数据为action dim，state shape为(num_stocks, window_size, num_features)
# action_dim = 11
# state_shape = (10, 50, 102)
short_term=10
# 知识蒸馏的相关参数,输入的未来时间窗口大小
future_window_size = 10
max_rule_consecutive_low = 3

# 关于SSM进入模型训练的流程
gmm_K = 2
use_gmm_in_train = True
inner_sample_lambda = 0.6
gmm_mix_lambda = 0.8

# Baseline
olmar_window_size=90
wmamr_window_size=90
markowitz_window_size=30
anticor_window_size=90

TRANSACTION_COST_RATE=5e-5

# Monitor reward: include transaction cost when switching
monitor_use_transaction_cost = True
monitor_transaction_cost_coef = 1.0

# ================= Reward scaling (global) =================
# 说明：环境里计算的都是 log-return / alpha 等“原始量纲”，为了让 PPO 的优势/价值有合适量级，
# 可以对各分支 reward 做统一缩放。
reward_scale_portfolio = 100.0
reward_scale_base = 100.0
reward_scale_outer = 100.0
reward_scale_inner = 2000.0
reward_scale_monitor = 100.0
reward_scale_controller = 100.0
controller_sup_coef = 0.0
controller_sup_horizon = 10
controller_algorithm = "pg"
controller_rollout_len = 400
controller_max_segments = 25
controller_pg_batch_windows = 4
controller_windows_per_epoch = 5
controller_start_stride_days = 40
controller_entropy_coef = 0.01
controller_aux_return_coef = 0.0
controller_aux_mdd_coef = 0.0
controller_aux_return_target_scale = 1.0
controller_aux_mdd_target_scale = 1.0
controller_mdd_coef = 2.0
controller_return_coef = 0.5
controller_count_min = 15
controller_count_max = 25
controller_count_penalty_coef = 0.5
controller_switch_coef = 0.0
controller_turnover_coef = 0.0
controller_check_stride_days = 1
inner_pred_coef = 0.0
inner_pred_target_scale = 1.0
outer_pred_coef = 0.1
inner_gate_reg_coef = 0.0
inner_use_topk = False
inner_feature_gate = False
inner_norm_mode = "legacy"
inner_train_fixed_episodes = True
inner_episode_len = 400
inner_train_episodes_per_epoch = 30
inner_train_start_stride_days = 120
inner_rollout_update_steps = 160
inner_ppo_epochs = 3
debug_inner_update_stats = False

# alphastock 参数
alphastcok = dict(
    look_back = 240, # 观察窗口
    step_size = 60, # 一个决策step的窗口大小
    num_epoch = 15, # 训练epoch
    batch_size = 128, # 训练batch大小
    num_steps = 12,
    model_param = dict(
        hidden_dim1 = 16,
        hidden_dim2 = 16,
        in_features = None,
        trade_num = 10, # 每次交易的股票数量

    )

)


# 特征提取/actor/critic的模型参数
model_dict=dict(
HierAgent = dict(
    model_param = dict(
        outer_actor = dict(
            hidden_dim1=16,
            hidden_dim2=16,
            in_features=None,
            trade_num=10,  # 每次交易的股票数量
        ),
        inner_actor = dict(
            num_pool=39,
            num_nodes=10,
            max_boundary=0.5,
            trade_num=2,  # 每次分别买卖的股票数量
            in_features=None,
            hidden_dim=16,
            dropout=0.2,
        ),
        outer_critic = dict(
            hidden_dim1=32,
            in_features=None,
        ),
        inner_critic = dict(
            hidden_dim1=32,
            in_features=None,
        ),
    )
),
)

# 输入env的数据参数，包括数据来源，特征，投资组合股票
dataset = dict(
    type = "PortfolioManagementDataset",
    ssm_data_path="Dataset/Nas100数据/feature_ssm", # 用于HRL训练的数据(参数+feature)
    raw_path="Dataset/Nas100数据/raw",
    feature_path = 'Dataset/Nas100数据/feature', # 用于普通强化学习训练的数据(防止过拟合的特征)
    ssm_feature = "Dataset/Nas100数据/ssm_feature",# 用于SSM训练的数据
    market_path="占位符",
    stocks_path ="utils/NAS100_pool.txt",
    prices_name = ['open', 'high', 'low', 'close'],
    features_name=[
        "adjopen",
        "adjhigh",
        "adjlow",
        "adjclose",
        "amount",
        "amp",
        "body",
    ],
 ssm_features = [
'open',
        'high',
        'low',
        'close',
        'adjfactor',
        'adjopen',
        'adjclose',
        'volume',
        'kmid2',
        'kup2',
        'klow',
        'klow2',
        'ksft2',
        'roc_5',
        'roc_10',
        'roc_20',
        'roc_30',
        'roc_60',
        'ma_5',
        'ma_10',
        'ma_20',
        'ma_30',
        'ma_60',
        'std_5',
        'std_10',
        'std_20',
        'std_30',
        'std_60',
        'beta_5',
        'beta_10',
        'beta_20',
        'beta_30',
        'beta_60',
        'max_5',
        'max_10',
        'max_20',
        'max_30',
        'max_60',
        'min_5',
        'min_10',
        'min_20',
        'min_30',
        'min_60',
        'qtlu_5',
        'qtlu_10',
        'qtlu_20',
        'qtlu_30',
        'qtlu_60',
        'qtld_5',
        'qtld_10',
        'qtld_20',
        'qtld_30',
        'qtld_60',
        'rank_5',
        'rank_10',
        'rank_20',
        'rank_30',
        'rank_60',
        'imax_5',
        'imax_10',
        'imax_20',
        'imax_30',
        'imax_60',
        'imin_5',
        'imin_10',
        'imin_20',
        'imin_30',
        'imin_60',
        'imxd_5',
        'imxd_10',
        'imxd_20',
        'imxd_30',
        'imxd_60',
        'cntp_5',
        'cntp_10',
        'cntp_20',
        'cntp_30',
        'cntp_60',
        'cntn_5',
        'cntn_10',
        'cntn_20',
        'cntn_30',
        'cntn_60',
        'cntd_5',
        'cntd_10',
        'cntd_20',
        'cntd_30',
        'cntd_60',
        'sump_5',
        'sump_10',
        'sump_20',
        'sump_30',
        'sump_60',
        'sumn_5',
        'sumn_10',
        'sumn_20',
        'sumn_30',
        'sumn_60',
        'sumd_5',
        'sumd_10',
        'sumd_20',
        'sumd_30',
        'sumd_60',
    ]
)


