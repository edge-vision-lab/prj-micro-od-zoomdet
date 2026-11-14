"""1組の正解と検出結果を TIDE にかけ、誤りの種類ごとの ΔAP を出す。

  閾値1つ  : --iou-thr 0.75（既定は 0.5）
  閾値の範囲: --iou-range 0.5 0.95 0.05（開始 終了 刻み）→ 各閾値の値と平均
  検出結果 : .json（CocoMetric の *.bbox.json）か .pkl（test.py --out の出力）
手法間の比較はしない（それぞれ実行して、出力した CSV を比べる）。

出力（{out-dir}/）:
  tide.csv       閾値ごとの値（範囲のときは最後に mean の行）
  settings.json  実行時の設定（あとで条件を確認するため）

置き場所: projects/Edge_poisson_asap/tools/run_tide.py（tide_lib.py と同じフォルダ）

# 判定 IoU：既定 0.5 / --iou-thr 0.75 / --iou-range 0.5 0.95 0.05（各閾値の行＋mean の行）
python projects/Edge_poisson_asap/tools/run_tide.py \
    --gt data_ann/VisDrone/my_val.json --det $OUT/results.bbox.json \
    [--iou-thr 0.75 | --iou-range 0.5 0.95 0.05] --name <条件がわかる名前>
# 出力：work_dirs/tide/{name}/tide.csv と settings.json（--name を変えないと上書き）
"""
import argparse
import csv
import json
import os

import numpy as np

from tide_lib import COLS, average_over_thresholds, eval_one_threshold, load_for_tide


def make_thresholds(args):
    """引数から、評価する IoU の閾値のリストを作る"""
    if args.iou_range is None:
        return [args.iou_thr]
    start, end, step = args.iou_range
    # 終了の値も含めるため、刻みの半分だけ先まで作り、小数の誤差を丸める
    return [float(t) for t in np.round(np.arange(start, end + step / 2, step), 4)]


def print_rows(rows):
    """表を表示する。rows: [(行の名前, {列名: 値}), ...]"""
    print('thr'.ljust(8) + ''.join(c.rjust(9) for c in COLS))
    for name, row in rows:
        print(name.ljust(8) + ''.join(f'{row[c]:9.2f}' for c in COLS))


def save_rows(rows, path):
    """表を CSV に保存する"""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['threshold'] + COLS)
        for name, row in rows:
            w.writerow([name] + [f'{row[c]:.3f}' for c in COLS])


def parse_args():
    """コマンドライン引数を読む"""
    p = argparse.ArgumentParser(description='1組の正解と検出結果を TIDE にかける')
    p.add_argument('--gt', required=True, help='正解の COCO 形式 json')
    p.add_argument('--det', required=True, help='検出結果（.json か .pkl）')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--iou-thr', type=float, default=0.5, help='正解とみなす IoU（閾値1つ）')
    g.add_argument('--iou-range', type=float, nargs=3, metavar=('START', 'END', 'STEP'),
                   help='閾値の範囲（例: 0.5 0.95 0.05）。各閾値の値と平均を出す')
    p.add_argument('--bg-thr', type=float, default=0.1, help='背景への誤検出とみなす IoU（標準は 0.1）')
    p.add_argument('--name', default=None, help='保存先の名前（既定は検出結果のフォルダ名）')
    p.add_argument('--out-dir', default=None, help='保存先（既定は work_dirs/tide/{name}）')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    name = args.name or os.path.basename(os.path.dirname(os.path.abspath(args.det)))
    out_dir = args.out_dir or os.path.join('work_dirs', 'tide', name)
    os.makedirs(out_dir, exist_ok=True)
    thresholds = make_thresholds(args)

    gt, preds = load_for_tide(args.gt, args.det, out_dir)
    per_thr = {}
    for t in thresholds:
        per_thr[t] = eval_one_threshold(gt, preds, t, args.bg_thr)
        print(f"IoU {t:.2f}: AP={per_thr[t]['AP']:.2f} Loc={per_thr[t]['Loc']:.2f}", flush=True)

    rows = [(f'{t:.2f}', row) for t, row in per_thr.items()]
    if len(thresholds) > 1:
        rows.append(('mean', average_over_thresholds(per_thr)))
    print()
    print_rows(rows)
    save_rows(rows, os.path.join(out_dir, 'tide.csv'))
    with open(os.path.join(out_dir, 'settings.json'), 'w') as f:
        json.dump({**vars(args), 'thresholds': thresholds}, f, indent=2, ensure_ascii=False)
    print(f'保存先: {out_dir}')