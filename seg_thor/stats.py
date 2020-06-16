import os
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt

stats_fields = ['epoch', 'batch', 'loss',
                'total_precision', 'precision_1', 'precision_2', 'precision_3', 'precision_4',
                'total_dice', 'dice_1', 'dice_2', 'dice_3', 'dice_4']
train_stats_path = '../data/{}_train_stats.csv'
eval_stats_path = '../data/{}_eval_stats.csv'

if __name__ == '__main__':
    parser = ArgumentParser()
    #parser.add_argument('operation', choices=['generate', 'frechet', 'inception'])
    parser.add_argument('-i', '--interval', default=100)
    parser.add_argument('-r', '--run_id', default=None, required=True)
    parser.add_argument('-o', '--output', default='../data/stats')
    args = parser.parse_args()


    run_id = args.run_id
    interval = int(args.interval)
    output_path = os.path.join(args.output, run_id)
    os.makedirs(output_path, exist_ok=True)

    cur_train_stats_path = train_stats_path.format(run_id)
    cur_eval_stats_path = eval_stats_path.format(run_id)

    train_df = pd.read_csv(cur_train_stats_path)
    eval_df = pd.read_csv(cur_eval_stats_path)

    train_batch_loss = train_df[train_df['total_precision', 'total_dice'].isnull().any(axis=1)]
    train_batch_loss = train_batch_loss.iloc[::interval]