"""
【メイン部分】1つの学習・推論結果について、次の図と表を出す。
  - IoU のヒストグラム（サイズで色分け）      : {name}_{subset}_hist_iou.png
  - スコアのヒストグラム（サイズで色分け）    : {name}_{subset}_hist_score.png
  - スコア（横）× IoU（縦）の散布図         : {name}_{subset}_scatter.png
  - 件数表（--iou-bounds と --score-bound を指定したときだけ）: {name}_count_table.csv と画面表示
「枠は出ているが位置の精度が悪いだけか」を見るために使う。手法どうしの比較は、
手法ごとに1回ずつ実行して図を並べて見る（軸範囲はどの図も 0〜1 で固定）。

入力は次のどちらか一方:
  (A) --pred       : 推論済みの *.bbox.json（test_evaluator.outfile_prefix で出したもの）
  (B) --checkpoint : 学習重み（.pth）。--config も必要。この場で test を回し、
                     out-dir/results.bbox.json を作ってから図を出す
図に使うデータは --subset で選ぶ（1つでも複数でもよい。複数なら図をそれぞれ出す）:
  all   : 重複を省いた全検出（重複の省き方は以前と同じ）
  match : そのうちクラス一致の検出だけ
件数表は表の中にクラス一致/不一致の列があるので、--subset によらず重複を省いた全検出で1つだけ出す。

置き場所: projects/Edge_poisson_asap/tools/plot_score_iou.py
使い方（workspace/ で実行）:
  (A) bbox.json から、全検出とクラス一致のみの両方の図と件数表を出す
     python projects/Edge_poisson_asap/tools/plot_score_iou.py \
         --pred work_dirs/eval_visdrone_uniform/results.bbox.json \
         --gt data_ann/VisDrone/my_val.json \
         --subset all match \
         --iou-bounds 0.1 0.5 --score-bound 0.3 \
         --out-dir work_dirs/score_iou_plots/uniform
  (B) 学習重みから（図はクラス一致のみ）
     python projects/Edge_poisson_asap/tools/plot_score_iou.py \
         --config <学習に使った設定ファイル>.py \
         --checkpoint work_dirs/yolox_tiny_visdrone_poisson_asap_e50/epoch_50.pth \
         --gt data_ann/VisDrone/my_val.json \
         --subset match \
         --out-dir work_dirs/score_iou_plots/poisson_asap
  変形後の座標で測るとき（サイズ区分は元画像の面積のまま）
     ... --pred <変形後の座標の results.bbox.json> \
         --gt data_ann/VisDrone_poisson_asap/val/annotations.json \
         --size-gt data_ann/VisDrone/my_val.json ...
"""
import argparse
import csv
import os
import sys

# tools/ から直接実行しても同じフォルダのモジュールを読み込めるようにする
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS_DIR)
from score_iou_figures import plot_histograms, plot_scatter  # noqa: E402
from score_iou_match import (SIZE_GROUPS, SUBSET_ALL, SUBSETS, count_table,  # noqa: E402
                             load_dets, load_gt, match_all, select_subset)

# 図のタイトルに付ける、データの選び方の説明
SUBSET_TITLES = {"all": "all detections (dedup)", "match": "class match only"}


def parse_args():
    """コマンドライン引数の読み取り"""
    p = argparse.ArgumentParser(description="1つの推論結果のスコア×IoU 散布図・IoU とスコアのヒストグラム・件数表")
    # 入力: bbox.json か学習重みのどちらか一方
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pred", help="推論済みの *.bbox.json（座標は --gt の正解とそろえる）")
    src.add_argument("--checkpoint", help="学習重み（.pth）。指定するとこの場で test を回す（--config 必須）")
    p.add_argument("--config", help="--checkpoint のときの設定ファイル（学習に使ったもの）")
    # 正解
    p.add_argument("--gt", required=True,
                   help="IoU を測る正解（例: data_ann/VisDrone/my_val.json）。検出結果と同じ座標系のもの")
    p.add_argument("--size-gt", default=None,
                   help="サイズ区分の面積を取る正解（省略時は --gt）。変形後の座標で測るときに元画像の json を指定")
    # 図の設定
    p.add_argument("--subset", nargs="+", choices=SUBSETS, default=[SUBSET_ALL],
                   help="図に使うデータ（1つでも複数でも可）。all = 重複を省いた全検出、"
                        "match = クラス一致の検出だけ（既定 all）")
    p.add_argument("--name", default=None,
                   help="図のタイトルとファイル名に使う名前（省略時は bbox.json のフォルダ名か設定ファイル名）")
    p.add_argument("--out-dir", required=True, help="図の保存先（--checkpoint のときは bbox.json もここに出る）")
    p.add_argument("--point-size", type=float, default=4.0, help="散布図の点の大きさ")
    p.add_argument("--alpha", type=float, default=0.2, help="散布図の点の不透明度（小さいほど透ける）")
    p.add_argument("--hist-scale", choices=["log", "linear"], default="log",
                   help="ヒストグラムの縦軸（既定 log）")
    # 件数表の設定
    p.add_argument("--iou-bounds", nargs="+", type=float, default=None,
                   help="件数表の IoU の帯の境界（例: 0.1 0.5）。省略すると件数表を出さない")
    p.add_argument("--score-bound", type=float, default=None,
                   help="件数表のスコア高/低の境界（例: 0.3）。--iou-bounds と一緒に指定する")
    args = p.parse_args()

    # 引数の組み合わせの確認
    if args.checkpoint is not None and args.config is None:
        p.error("--checkpoint を使うときは --config も指定してください")
    if args.pred is not None and args.config is not None:
        p.error("--config は --checkpoint と一緒に使います（--pred のときは不要です）")
    if (args.iou_bounds is None) != (args.score_bound is None):
        p.error("--iou-bounds と --score-bound は両方指定するか、両方省略してください")
    if args.iou_bounds is not None:
        b = args.iou_bounds
        if any(not 0 < v < 1 for v in b) or b != sorted(b):
            p.error("--iou-bounds は 0 と 1 の間で小さい順に指定してください")
    # 同じ subset を2回書かれても図は1回だけ出す（順番は書かれた順のまま）
    args.subset = list(dict.fromkeys(args.subset))
    return args


def default_name(args):
    """--name を省略したときの名前（bbox.json の入ったフォルダ名、または設定ファイル名）"""
    if args.pred is not None:
        return os.path.basename(os.path.dirname(os.path.abspath(args.pred)))
    return os.path.splitext(os.path.basename(args.config))[0]


def run_inference(config_path, checkpoint_path, out_dir):
    """
    学習重みで val（設定の test_dataloader）の推論を行い、out_dir/results.bbox.json を作ってそのパスを返す。
    mmdet の tools/test.py と同じく Runner.test() を使い、test_evaluator.outfile_prefix だけ書き換える。
    """
    # 新手法の設定は custom_imports で edge_poisson_asap を読み込むので、
    # PYTHONPATH を設定しなくても読めるよう projects/Edge_poisson_asap を探索パスに加える
    sys.path.insert(0, os.path.dirname(TOOLS_DIR))
    # mmengine は重いので、学習重みから推論するときだけ読み込む
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(config_path)
    cfg.load_from = checkpoint_path
    cfg.work_dir = os.path.join(out_dir, "test_work_dir")    # ログの置き場所
    if isinstance(cfg.test_evaluator, (list, tuple)):
        raise ValueError("test_evaluator が複数ある設定には対応していません（1つにしてください）")
    # 評価クラスに検出結果を json で書き出させる（out_dir/results.bbox.json になる）
    prefix = os.path.join(os.path.abspath(out_dir), "results")
    cfg.test_evaluator.outfile_prefix = prefix
    Runner.from_cfg(cfg).test()
    return prefix + ".bbox.json"


def print_stats(name, st):
    """割り当ての集計を画面に表示する（score_thr の確認や件数の把握用）"""
    print(f"[{name}] 画像 {st['n_images']} 枚, 検出 {st['n_det']} 個, "
          f"重複として省いた検出 {st['n_dup']} 個, 残した検出 {st['n_kept']} 個"
          f"（うちクラス一致 {st['n_match']} 個）, "
          f"1枚あたりの最大検出数 {st['max_per_img']}, 最小スコア {st['min_score']:.4f}")


def write_tables(matched, name, iou_bounds, score_bound, path):
    """
    件数表をサイズの区切り（all / small / medium / large）ごとに作り、1つの CSV に保存して画面にも表示する
    （以前の write_tables を1手法用にし、サイズの列を足したもの）
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "size", "score_band", "iou_band", "class", "count"])
        for size, size_name in SIZE_GROUPS:
            rows = count_table(matched, iou_bounds, score_bound, size)
            for r in rows:
                w.writerow([name, size_name, *r])
            print_table(name, size_name, rows)


def print_table(name, size_name, rows):
    """件数表を「スコアの帯 × IoU の帯」の行、「クラス一致/不一致/計」の列で表示する"""
    print(f"\n=== 件数表: {name} / サイズ {size_name} ===")
    print(f"{'score':<12}{'IoU':<16}{'match':>10}{'mismatch':>10}{'total':>10}")
    # rows は (スコアの帯, IoU の帯) ごとに 一致 → 不一致 の順で2行ずつ並んでいる
    for i in range(0, len(rows), 2):
        s_name, i_name, _, n_match = rows[i]
        n_mis = rows[i + 1][3]
        print(f"{s_name:<12}{i_name:<16}{n_match:>10}{n_mis:>10}{n_match + n_mis:>10}")
        

def save_figures(matched, subset, name, args):
    """1つの subset について、散布図と2つのヒストグラムを保存し、保存したファイル名の頭を返す"""
    data = select_subset(matched, subset)
    title = f"{name} [{SUBSET_TITLES[subset]}]"
    prefix = os.path.join(args.out_dir, f"{name}_{subset}")
    plot_scatter(data, title, prefix + "_scatter.png", args.point_size, args.alpha)
    plot_histograms(data, title, prefix + "_hist_iou.png", prefix + "_hist_score.png",
                    log_y=(args.hist_scale == "log"))
    return prefix


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 入力の準備: 学習重みが指定されたら先に推論して bbox.json を作る
    pred_path = args.pred if args.pred is not None else \
        run_inference(args.config, args.checkpoint, args.out_dir)
    name = args.name or default_name(args)

    # 集計: 検出と正解を対応付ける（重複の除去まで。subset での絞り込みは図ごとに行う）
    gt = load_gt(args.gt, size_gt_path=args.size_gt)
    matched, stats = match_all(gt, load_dets(pred_path))
    print_stats(name, stats)

    # 描画: 指定された subset ごとに3つの図を保存する
    prefixes = [save_figures(matched, subset, name, args) for subset in args.subset]

    # 件数表: 境界が指定されたときだけ、重複を省いた全検出で出す
    if args.iou_bounds is not None:
        write_tables(matched, name, args.iou_bounds, args.score_bound,
                     os.path.join(args.out_dir, f"{name}_count_table.csv"))

    print("\n注記: my_val.json には VisDrone の無視領域（ignored regions / others）が入っていないため、"
          "無視領域への検出も相手なし（IoU = 0）として数えている")
    for prefix in prefixes:
        print(f"保存先: {prefix}_{{scatter, hist_iou, hist_score}}.png")


if __name__ == "__main__":
    main()