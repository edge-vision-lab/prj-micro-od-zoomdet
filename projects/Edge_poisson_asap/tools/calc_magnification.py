"""calc_magnification.py
変形前後の bbox（COCO 形式の json）から物体の拡大率を計算し、
全体・サイズ区分ごとの統計と最大値を表示する。必要ならヒストグラムと CSV も保存する。

使い方の例（変形後データのフォルダを指定。枠サイズと元の正解は settings.json から読む）:
    python projects/Edge_poisson_asap/tools/calc_magnification.py \
        --warp-dir data_ann/VisDrone_poisson_asap_mean_3/val \
        --hist mag_hist.png --csv mag_per_object.csv \
        --out-dir work_dirs/eval_visdrone_poisson_asap_mean3 \
        --json-out fact.json

settings.json がない古いフォルダでは --orig と --frame を指定する。
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')  # plt.show() が使えない環境なので画像保存用のバックエンドにする
import matplotlib.pyplot as plt
import numpy as np

from magnification_lib import METRICS, SIZE_NAMES, compute_records, summarize_records

# 変形後データのフォルダ内のファイル名
SETTINGS_NAME = 'settings.json'    # 変形時の設定（枠サイズ box、元の正解 ann など）
ANNOTATION_NAME = 'annotations.json'  # 変形後の正解

# ヒストグラムの横軸の範囲（最小, 最大）。範囲を変えたいときはここを書き換える
# （実行時に --hist-range 0.5 3.0 のように一時的に変えることもできる）
HIST_RANGE = (0, 4)

# 表示内容を保存するテキストファイルの既定名
SUMMARY_TXT_NAME = 'mag_summary.txt'

# 表示用の指標名
METRIC_LABELS = {'area_ratio': '面積倍率', 'length_ratio': '長さ倍率(辺)'}


def parse_args():
    """コマンドライン引数を読む。"""
    p = argparse.ArgumentParser(description='物体ごとの拡大率を計算する')
    p.add_argument('--warp-dir', required=True,
                   help=f'変形後データのフォルダ（{ANNOTATION_NAME} と {SETTINGS_NAME} がある場所）')
    p.add_argument('--orig', default=None,
                   help=f'元座標の COCO 形式 json。省略時は {SETTINGS_NAME} の ann を使う')
    p.add_argument('--baseline', choices=['frame', 'image', 'none'], default='frame',
                   help='基準: frame=枠に収めた一様縮小 / image=json の画像サイズ比 / none=元座標との生の比')
    p.add_argument('--frame', type=int, nargs=2, default=None, metavar=('W', 'H'),
                   help=f'枠サイズ。{SETTINGS_NAME} がないフォルダのときだけ使う（ある場合は box と一致を確認）')
    p.add_argument('--size-source', choices=['area', 'bbox'], default='area',
                   help='サイズ区分に使う面積: area=json の area フィールド / bbox=w*h')
    p.add_argument('--keep-crowd', action='store_true', help='iscrowd=1 も含める')
    p.add_argument('--hist', default=None, help='ヒストグラムの保存先（指定時のみ作成）')
    p.add_argument('--hist-metric', choices=list(METRICS), default='area_ratio',
                   help='ヒストグラムにする指標')
    p.add_argument('--hist-range', type=float, nargs=2, default=list(HIST_RANGE),
                   metavar=('MIN', 'MAX'),
                   help=f'ヒストグラムの横軸の範囲（既定 {HIST_RANGE[0]} {HIST_RANGE[1]}）')
    p.add_argument('--bins', type=int, default=50, help='ヒストグラムの区間数')
    p.add_argument('--log-x', action='store_true', help='横軸を対数にする（区間も対数等間隔）')
    p.add_argument('--csv', default=None, help='物体ごとの結果を保存する CSV（任意）')
    p.add_argument('--txt', default=SUMMARY_TXT_NAME,
                   help=f'表示内容（統計表など）を保存するテキストファイル（既定 {SUMMARY_TXT_NAME}）')
    p.add_argument('--no-txt', action='store_true', help='テキストファイルを保存しない')
    p.add_argument('--json-out', default=None, help='統計をまとめて保存する json（任意）')
    p.add_argument('--out-dir', default=None,
                   help='出力先ディレクトリ（なければ作成）。--hist/--csv/--json-out の相対パスはこの中に置く')
    return p.parse_args()


class TextLog:
    """画面に表示しつつ、同じ内容を貯めておいて最後にテキストファイルへ書き出す。"""

    def __init__(self):
        self.lines = []  # これまでに表示した行

    def __call__(self, msg=''):
        """1 行（改行を含んでもよい）を表示して記録する。print の代わりに使う。"""
        print(msg)
        self.lines.append(msg)

    def save(self, path):
        """記録した内容をテキストファイルに保存する。"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.lines) + '\n')


def print_summary(stats, argmax, metric, log):
    """統計表と最大拡大率の物体を表示する（log 経由なのでテキストにも残る）。"""
    log(f'\n=== {METRIC_LABELS[metric]} ({metric}) ===')
    log(f'{"size":<7}{"count":>8}{"mean":>9}{"geo_mean":>10}{"median":>9}'
        f'{"std":>9}{"min":>9}{"max":>9}')
    for k in ('all',) + SIZE_NAMES:
        s = stats[k]
        if s['count'] == 0:
            log(f'{k:<7}{0:>8}')
            continue
        log(f'{k:<7}{s["count"]:>8}{s["mean"]:>9.3f}{s["geo_mean"]:>10.3f}{s["median"]:>9.3f}'
            f'{s["std"]:>9.3f}{s["min"]:>9.3f}{s["max"]:>9.3f}')
    log('最大拡大率の物体:')
    for k, r in argmax.items():
        log(f'  {k:<7} ann_id={r["ann_id"]} image_id={r["image_id"]} '
            f'元bbox={r["w"]:.1f}x{r["h"]:.1f} 変形後={r["w_warp"]:.1f}x{r["h_warp"]:.1f} '
            f'値={r[metric]:.3f}')

def resolve_settings_path(value, settings_out_dir, warp_dir):
    """settings.json 内の相対パスを、今の作業ディレクトリから使えるパスに直す。

    settings.json のパスは変形スクリプトを実行した場所からの相対パスで書かれているため、
    そのまま使うと実行場所によってずれる。そこで settings の out_dir（＝warp_dir 自身）
    からの相対位置に直してから warp_dir につなげる。
    例: ann='../../data_ann/VisDrone/my_val.json', out_dir='../../data_ann/X/val'
        → '../../VisDrone/my_val.json' を warp_dir につなげる → data_ann/VisDrone/my_val.json
    """
    if os.path.isabs(value):
        return value
    rel_from_out = os.path.relpath(value, settings_out_dir)  # out_dir から見た相対パス
    return os.path.normpath(os.path.join(warp_dir, rel_from_out))


def resolve_inputs(args):
    """--warp-dir と settings.json から、(元の json, 変形後の json, 枠サイズ, 設定) を決める。

    枠サイズは settings.json の box を正とし、--frame はなかったときの代わりにだけ使う。
    両方あって値が違えば、取り違えの可能性が高いので止める。
    """
    warp_json = os.path.join(args.warp_dir, ANNOTATION_NAME)  # 変形後の正解
    if not os.path.isfile(warp_json):
        raise FileNotFoundError(f'{warp_json} がありません')
    settings_path = os.path.join(args.warp_dir, SETTINGS_NAME)
    settings = None  # settings.json の中身（ない場合は None）
    if os.path.isfile(settings_path):
        with open(settings_path) as f:
            settings = json.load(f)

    # 枠サイズを決める
    if settings is not None and 'box' in settings:
        frame = tuple(settings['box'])  # [幅, 高さ] として扱う
        if args.frame is not None and tuple(args.frame) != frame:
            raise ValueError(f'--frame {args.frame} と {SETTINGS_NAME} の box {list(frame)} が一致しません')
    elif args.frame is not None:
        frame = tuple(args.frame)
    else:
        raise ValueError(f'{SETTINGS_NAME} に box がないので --frame を指定してください')

    # 元の正解を決める（--orig 指定が優先、なければ settings の ann）
    if args.orig is not None:
        orig_json = args.orig
    elif settings is not None and 'ann' in settings:
        orig_json = resolve_settings_path(settings['ann'], settings.get('out_dir', args.warp_dir),
                                          args.warp_dir)
    else:
        raise ValueError(f'{SETTINGS_NAME} に ann がないので --orig を指定してください')
    if not os.path.isfile(orig_json):
        raise FileNotFoundError(f'元の正解 {orig_json} がありません（--orig で指定してください）')
    return orig_json, warp_json, frame, settings


def resolve_out_path(path, out_dir):
    """出力ファイルのパスを決める。

    out_dir が指定され、path が相対パスなら out_dir の中に置く。
    絶対パスや out_dir 未指定のときは path をそのまま使う。
    """
    if path is None or out_dir is None or os.path.isabs(path):
        return path
    return os.path.join(out_dir, path)

def save_csv(records, path):
    """物体ごとの結果を CSV に保存する。"""
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def save_histogram(records, metric, path, bins, log_x, hist_range, log):
    """サイズ区分ごとに色を分けた積み上げヒストグラムを、固定の横軸範囲で保存する。

    hist_range=(最小, 最大) の外にある値は描かず、その件数をタイトルと標準出力に出す。
    """
    lo, hi = hist_range  # 横軸の両端（固定値）
    if not lo < hi:
        raise ValueError(f'ヒストグラムの範囲が不正です: {hist_range}')
    if log_x and lo <= 0:
        raise ValueError('対数軸では範囲の最小値を 0 より大きくしてください')
    if log_x:
        edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1)  # 対数等間隔の区間
    else:
        edges = np.linspace(lo, hi, bins + 1)  # 等間隔の区間
    values = np.array([r[metric] for r in records])  # 全物体の値（範囲外の件数を数える用）
    n_below = int((values < lo).sum())  # 範囲より小さくて描かれない物体の数
    n_above = int((values > hi).sum())  # 範囲より大きくて描かれない物体の数
    # サイズ区分ごとの値のリスト
    per_size = [[r[metric] for r in records if r['size'] == n] for n in SIZE_NAMES]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(per_size, bins=edges, stacked=True, label=list(SIZE_NAMES))
    ax.axvline(1.0, color='k', linestyle='--', linewidth=1, label='1.0 (= uniform)')  # 日本語フォントがない環境でも文字化けしないよう英語
    if log_x:
        ax.set_xscale('log')
    ax.set_xlim(lo, hi)  # 手法間で見比べられるよう横軸を固定する
    ax.set_xlabel(metric)
    ax.set_ylabel('count')
    ax.set_title(f'{metric} (n={len(records)}, out of range: <{lo} {n_below}, >{hi} {n_above})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if n_below or n_above:
        log(f'ヒストグラムの範囲外（描画なし）: {lo} 未満 {n_below} 件、{hi} 超 {n_above} 件')


def main():
    """拡大率を計算して表示・保存する。"""
    args = parse_args()
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)  # 出力先がなければ作る
    # 各出力ファイルの保存先を出力ディレクトリに合わせて決める
    hist_path = resolve_out_path(args.hist, args.out_dir)
    csv_path = resolve_out_path(args.csv, args.out_dir)
    json_path = resolve_out_path(args.json_out, args.out_dir)
    # テキスト保存先（--no-txt なら保存しない）
    txt_path = None if args.no_txt else resolve_out_path(args.txt, args.out_dir)
    log = TextLog()  # 表示とテキスト保存を兼ねる
    orig_json, warp_json, frame, settings = resolve_inputs(args)
    # 何を使って計算したかを最初に表示する（取り違えの確認用）
    log(f'元の正解  : {orig_json}')
    log(f'変形後正解: {warp_json}')
    log(f'枠サイズ  : {frame[0]}x{frame[1]}' + ('' if settings else '（--frame で指定）'))
    records, skipped = compute_records(
        orig_json, warp_json, baseline=args.baseline, frame=frame,
        size_source=args.size_source, skip_crowd=not args.keep_crowd)
    log(f'対象物体数: {len(records)}  除外: {skipped}  baseline={args.baseline}')
    if not records:
        log('対象となる物体がありません（id の対応付けを確認してください）')
        if txt_path:
            log.save(txt_path)
        return
    # json 出力用のまとめ（使った入力と変形時の設定も残す）
    out = {'orig': orig_json, 'warp': warp_json, 'frame': list(frame),
           'warp_settings': settings, 'skipped': skipped, 'baseline': args.baseline}
    for metric in METRICS:
        stats, argmax = summarize_records(records, metric)
        print_summary(stats, argmax, metric, log)
        out[metric] = {'stats': stats,
                       'max_object': {k: r['ann_id'] for k, r in argmax.items()}}
    if hist_path:
        save_histogram(records, args.hist_metric, hist_path, args.bins, args.log_x,
                       tuple(args.hist_range), log)
        log(f'ヒストグラムを保存: {hist_path}')
    if csv_path:
        save_csv(records, csv_path)
        log(f'CSV を保存: {csv_path}')
    if json_path:
        with open(json_path, 'w') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        log(f'統計を保存: {json_path}')
    if txt_path:
        log(f'表示内容を保存: {txt_path}')
        log.save(txt_path)  # 最後の行も含めて書き出す

if __name__ == '__main__':
    main()