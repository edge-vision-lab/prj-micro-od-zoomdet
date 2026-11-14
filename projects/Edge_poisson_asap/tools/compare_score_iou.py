"""
【メイン部分・手法比較】複数の *.bbox.json（最大5個）を、同じグラフに重ねて比べる。
出す図は4枚（どれもサイズの色分けなし、色 = 手法）:
  - compare_hist_iou_all.png     : 重複を省いた全検出の IoU のヒストグラム
  - compare_hist_iou_match.png   : クラス一致の検出だけの IoU のヒストグラム
  - compare_hist_score_all.png   : 重複を省いた全検出のスコアのヒストグラム
  - compare_hist_score_match.png : クラス一致の検出だけのスコアのヒストグラム
集計（score_iou_match.py）と描画（score_iou_figures.py）は1結果用のコードと共通で使う。
重複の省き方・クラス一致の決め方は plot_score_iou.py と同じ。

置き場所: projects/Edge_poisson_asap/tools/compare_score_iou.py
使い方（workspace/ で実行）:
  python projects/Edge_poisson_asap/tools/compare_score_iou.py \
      --preds work_dirs/eval_visdrone_uniform/results.bbox.json \
              work_dirs/eval_visdrone_poisson_asap_e50/results.bbox.json \
      --names uniform poisson_asap \
      --gt data_ann/VisDrone/my_val.json \
      --out-dir work_dirs/score_iou_compare
  変形後の座標の結果を混ぜるとき（手法ごとに IoU を測る正解を指定する）
      ... --gts data_ann/VisDrone/my_val.json \
                data_ann/VisDrone/my_val.json \
                data_ann/VisDrone_poisson_asap/val/annotations.json ...
"""
import argparse
import os
import sys

# tools/ から直接実行しても同じフォルダのモジュールを読み込めるようにする
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_iou_figures import METHOD_COLORS, plot_compare_histogram  # noqa: E402
from score_iou_match import (SUBSET_ALL, SUBSET_MATCH, load_dets, load_gt,  # noqa: E402
                             match_all, select_subset)

# 比べられる手法の数の上限（描画側で用意している色の数）
MAX_METHODS = len(METHOD_COLORS)

# 作る図の一覧: (図に使うデータ, 値の名前, 横軸のラベル, タイトルの説明)
FIGURES = [
    (SUBSET_ALL, "iou", "IoU", "IoU, all detections (dedup)"),
    (SUBSET_MATCH, "iou", "IoU", "IoU, class match only"),
    (SUBSET_ALL, "score", "confidence score", "score, all detections (dedup)"),
    (SUBSET_MATCH, "score", "confidence score", "score, class match only"),
]


def parse_args():
    """コマンドライン引数の読み取り"""
    p = argparse.ArgumentParser(description="複数手法の IoU・スコアのヒストグラムを重ねて比べる")
    p.add_argument("--preds", nargs="+", required=True,
                   help=f"手法ごとの *.bbox.json（{MAX_METHODS} 個まで）")
    p.add_argument("--names", nargs="+", default=None,
                   help="手法名（--preds と同じ順・同じ数）。省略時は各 bbox.json のフォルダ名")
    p.add_argument("--gt", required=True,
                   help="IoU を測る正解（例: data_ann/VisDrone/my_val.json）。--gts を省略したら全手法でこれを使う")
    p.add_argument("--gts", nargs="+", default=None,
                   help="手法ごとに IoU を測る正解（--preds と同じ順・同じ数）。変形後の座標の結果を混ぜるときに使う")
    p.add_argument("--out-dir", required=True, help="図の保存先")
    p.add_argument("--hist-scale", choices=["log", "linear"], default="log",
                   help="ヒストグラムの縦軸（既定 log）")
    args = p.parse_args()

    # 引数の組み合わせの確認
    if len(args.preds) > MAX_METHODS:
        p.error(f"--preds は {MAX_METHODS} 個までにしてください")
    if args.names is not None and len(args.names) != len(args.preds):
        p.error("--names は --preds と同じ数だけ指定してください")
    if args.gts is not None and len(args.gts) != len(args.preds):
        p.error("--gts は --preds と同じ数だけ指定してください")
    if args.names is None:
        args.names = [os.path.basename(os.path.dirname(os.path.abspath(x))) for x in args.preds]
    if len(set(args.names)) != len(args.names):
        p.error(f"手法名が重複しています: {args.names}（--names で別の名前を付けてください）")
    return args


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 集計: 手法ごとに検出と正解を対応付ける。同じ正解 json は1回だけ読む
    gt_paths = args.gts if args.gts is not None else [args.gt] * len(args.preds)
    gt_cache = {path: load_gt(path) for path in dict.fromkeys(gt_paths)}
    matched = []
    for pred_path, gt_path, name in zip(args.preds, gt_paths, args.names):
        m, st = match_all(gt_cache[gt_path], load_dets(pred_path))
        matched.append(m)
        print(f"[{name}] 検出 {st['n_det']} 個, 重複として省いた検出 {st['n_dup']} 個, "
              f"残した検出 {st['n_kept']} 個（うちクラス一致 {st['n_match']} 個）, "
              f"最小スコア {st['min_score']:.4f}")

    # 描画: 4枚の図（全検出/クラス一致のみ × IoU/スコア）を保存する
    log_y = args.hist_scale == "log"
    for subset, key, xlabel, desc in FIGURES:
        datas = [select_subset(m, subset) for m in matched]
        path = os.path.join(args.out_dir, f"compare_hist_{key}_{subset}.png")
        plot_compare_histogram(datas, args.names, key, xlabel, desc, path, log_y)
        print(f"保存先: {path}")


if __name__ == "__main__":
    main()