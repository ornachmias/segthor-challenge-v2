import os
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt

stats_fields = ['epoch', 'batch', 'loss',
                'total_precision', 'precision_1', 'precision_2', 'precision_3', 'precision_4',
                'total_dice', 'dice_1', 'dice_2', 'dice_3', 'dice_4']
train_stats_path = '../data/{}_train_stats.csv'
eval_stats_path = '../data/{}_eval_stats.csv'


def draw_batch_loss(df, interval, path, title):
    batch_loss = df[df[['total_precision', 'total_dice']].isnull().any(axis=1)][['step', 'loss']]
    batch_loss['loss'] = batch_loss['loss'].rolling(interval).mean()
    batch_loss = batch_loss.iloc[::interval]
    plot = batch_loss.plot(x='step', y='loss', kind='line', title=title)
    plot.get_figure().savefig(path)


def draw_epoch_loss(df, path, title):
    epoch_loss = df[df[['total_precision', 'total_dice']].notnull().any(axis=1)][['epoch', 'loss']]
    plot = epoch_loss.plot(x='epoch', y='loss', kind='line', title=title)
    plot.get_figure().savefig(path)


def draw_precision(df, path, title):
    precision = df[df['total_precision'].notnull()]
    plot = precision.plot(x='epoch',
                         y=['total_precision', 'precision_1', 'precision_2', 'precision_3', 'precision_4'],
                         kind='line', title=title)
    plot.get_figure().savefig(path)


def draw_dice(df, path, title):
    precision = df[df['total_dice'].notnull()]
    plot = precision.plot(x='epoch',
                         y=['total_dice', 'dice_1', 'dice_2', 'dice_3', 'dice_4'],
                         kind='line', title=title)
    plot.get_figure().savefig(path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--interval', default=10)
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
    train_df.insert(0, 'step', range(len(train_df)))
    draw_batch_loss(train_df, interval, os.path.join(output_path, 'train_batch_loss.png'), 'Train Batch Loss')
    draw_epoch_loss(train_df, os.path.join(output_path, 'train_epoch_loss.png'), 'Train Epoch Loss')
    draw_precision(train_df, os.path.join(output_path, 'train_precision.png'), 'Train Precision')

    eval_df = pd.read_csv(cur_eval_stats_path)
    eval_df.insert(0, 'step', range(len(eval_df)))
    draw_batch_loss(eval_df, interval, os.path.join(output_path, 'eval_batch_loss.png'), 'Eval Batch Loss')
    draw_epoch_loss(eval_df, os.path.join(output_path, 'eval_epoch_loss.png'), 'Eval Epoch Loss')
    draw_precision(eval_df, os.path.join(output_path, 'eval_precision.png'), 'Eval Precision')
    draw_dice(eval_df, os.path.join(output_path, 'eval_dice.png'), 'Eval Dice')