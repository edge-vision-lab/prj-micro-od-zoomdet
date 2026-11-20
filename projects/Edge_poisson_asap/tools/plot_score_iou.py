"""
【メイン部分】1つの学習・推論結果について、次の図と表を出す。
  - IoU のヒストグラム（サイズで色分け）      : {name}_{subset}_hist_iou.png
  - スコアのヒストグラム（サイズで色分け）    : {name}_{subset}_hist_score.png
  - スコア（横）× IoU（縦）の散布図         : {name}_{subset}_scatter.png
  - 件数表（画面表示と {name}_summary.csv）
「枠は出ているが位置の精度が悪いだけか」を見るために使う。手法どうしの比較は、
手法ごとに1回ずつ実行して図を並べて見る（軸範囲はどの図も 0〜1 で固定）。

入力は次のどちらか一方:
  (A) --pred       : 推論済みの *.bbox.json（test_evaluator.outfile_prefix で出したもの）
  (B) --checkpoint : 学習重み（.pth）。--config も必要。この場で推論してから図を出す。
                     検出結果の座標は --coords で選ぶ:
                       orig     : test を回し（Runner.test()、tools/test.py と同じ）、
                                  out-dir/results.bbox.json を作る（既定。以前と同じ）。
                                  非一様縮小の設定では評価クラスが OriginalCoordsCocoMetric なので元座標になる
                       deformed : 変形後の画像（--warp-dir/images）をそのまま推論し（deformed_predict.run_detection）、
                                  out-dir/pred_deformed.json を作る（変形後の座標。非一様縮小の設定で使う）。
                                  --gt を省略すると --warp-dir/annotations.json（変形後の座標の正解）を使う
                     --score-thr を付けると model.test_cfg.score_thr をその値にする（どちらの座標でも同じ）
図に使うデータは --subset で選ぶ（1つでも複数でもよい。複数なら図をそれぞれ出す）:
  all   : 重複を省いた全検出（重複の省き方は以前と同じ）
  match : そのうちクラス一致の検出だけ
件数表は表の中にクラス正解/不正解の行があるので、--subset によらず重複を省いた全検出から作る。
項目は「検出枠数・クラス正解/不正解・IoU 閾値での内訳・クラス正解の IoU の平均と中央値」で、
「全検出 / score>=t（--summary-score-thrs）」×「all / small / medium / large」ごとに出す。
相手なし（IoU = 0）の検出はサイズが決まらないので all にだけ入る。

置き場所: projects/Edge_poisson_asap/tools/plot_score_iou.py
使い方（workspace/ で実行）:
  (A) bbox.json から、全検出とクラス一致のみの両方の図と件数表を出す
     python projects/Edge_poisson_asap/tools/plot_score_iou.py \
         --pred work_dirs/eval_visdrone_uniform_e50/results.bbox.json \
         --gt data_ann/VisDrone/my_val.json \
         --subset all match \
         --out-dir work_dirs/score_iou_plots/uniform
  (B) 学習重みから（図はクラス一致のみ）
     python projects/Edge_poisson_asap/tools/plot_score_iou.py \
         --config <学習に使った設定ファイル>.py \
         --checkpoint work_dirs/yolox_tiny_visdrone_poisson_asap_e50/epoch_50.pth \
         --gt data_ann/VisDrone/my_val.json \
         --subset match \
         --out-dir work_dirs/score_iou_plots/poisson_asap
  変形後の座標で測るとき（サイズ区分は元画像の面積のまま）
     ... --pred <変形後の座標の bbox.json（前に (B) deformed で作った pred_deformed.json など）> \
         --gt data_ann/VisDrone_poisson_asap/val/annotations.json \
         --size-gt data_ann/VisDrone/my_val.json ...
  (B) 学習重みから、変形後の座標で（サイズ区分は元画像の面積のまま）
     python projects/Edge_poisson_asap/tools/plot_score_iou.py \
         --config projects/Edge_poisson_asap/configs/yolox_tiny_visdrone_poisson_asap.py \
         --checkpoint work_dirs/yolox_tiny_visdrone_poisson_asap_e50/epoch_50.pth \
         --coords deformed --warp-dir data_ann/VisDrone_poisson_asap/val \
         --size-gt data_ann/VisDrone/my_val.json \
         --score-thr 0.01 --subset all match \
         --out-dir work_dirs/score_iou_plots/poisson_asap_deformed
"""
import argparse
import csv
import os
import sys

# tools/ から直接実行しても同じフォルダのモジュールを読み込めるようにする
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS_DIR)
from eval_common import text_width  # noqa: E402
from score_iou_figures import plot_histograms, plot_scatter  # noqa: E402
from score_iou_match import (SUBSET_ALL, SUBSETS, build_summaries,  # noqa: E402
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
    p.add_argument("--coords", choices=["orig", "deformed"], default="orig",
                   help="--checkpoint のときの検出結果の座標。orig = test を回す（既定・以前と同じ）、"
                        "deformed = 変形後の画像をそのまま推論する（--warp-dir が必要）")
    p.add_argument("--warp-dir", default=None,
                   help="--coords deformed のときの変形後のデータ（images/ と annotations.json がある所）")
    p.add_argument("--score-thr", type=float, default=None,
                   help="--checkpoint のときの model.test_cfg.score_thr（省略時は設定ファイルの値）。"
                        "tools/test.py にそろえるなら 0.01")
    p.add_argument("--device", default="cuda:0", help="--coords deformed のときの推論に使う装置")
    # 正解
    p.add_argument("--gt", default=None,
                   help="IoU を測る正解（例: data_ann/VisDrone/my_val.json）。検出結果と同じ座標系のもの。"
                        "--coords deformed のときだけ省略でき、--warp-dir/annotations.json になる")
    p.add_argument("--size-gt", default=None,
                   help="サイズ区分の面積を取る正解（省略時は --gt）。変形後の座標で測るときに元画像の json を指定")
    # 図の設定
    p.add_argument("--subset", nargs="+", choices=SUBSETS, default=[SUBSET_ALL],
                   help="図に使うデータ（1つでも複数でも可）。all = 重複を省いた全検出、"
                        "match = クラス一致の検出だけ（既定 all）")
    p.add_argument("--name", default=None,
                   help="図のタイトルとファイル名に使う名前（省略時は bbox.json のフォルダ名か設定ファイル名）")
    p.add_argument("--out-dir", required=True,
                   help="図の保存先（--checkpoint のときは検出結果の json もここに出る）")
    p.add_argument("--point-size", type=float, default=4.0, help="散布図の点の大きさ")
    p.add_argument("--alpha", type=float, default=0.2, help="散布図の点の不透明度（小さいほど透ける）")
    p.add_argument("--hist-scale", choices=["log", "linear"], default="log",
                   help="ヒストグラムの縦軸（既定 log）")
    # 件数表の設定
    p.add_argument("--iou-thr", type=float, default=0.5,
                   help="件数表で「IoU が高い/低い」を分ける閾値（既定 0.5）")
    p.add_argument("--summary-score-thrs", nargs="*", type=float, default=[0.3],
                   help="件数表をスコアで絞った版も出すときの閾値（複数可、既定 0.3）。"
                        "何も書かなければ全検出の表だけ")
    args = p.parse_args()

    # 引数の組み合わせの確認
    if args.checkpoint is not None and args.config is None:
        p.error("--checkpoint を使うときは --config も指定してください")
    if args.pred is not None and args.config is not None:
        p.error("--config は --checkpoint と一緒に使います（--pred のときは不要です）")
    deformed = args.checkpoint is not None and args.coords == "deformed"
    if args.pred is not None and (args.coords != "orig" or args.warp_dir is not None):
        p.error("--coords と --warp-dir は --checkpoint と一緒に使います（--pred のときは --gt で座標を合わせます）")
    if deformed and args.warp_dir is None:
        p.error("--coords deformed には --warp-dir が必要です")
    if args.gt is None:
        if not deformed:
            p.error("--gt を指定してください（省略できるのは --coords deformed のときだけです）")
        args.gt = os.path.join(args.warp_dir, "annotations.json")   # 変形後の座標の正解
    if deformed and args.size_gt is None:
        # 変形後の正解の面積のままだと、一様縮小とサイズ区分がそろわないことを知らせる（止めはしない）
        print("注意: --size-gt がないので、サイズ区分は変形後の面積で決めます"
              "（元画像の面積にするなら --size-gt に元の正解を指定）")
    # 同じ subset を2回書かれても図は1回だけ出す（順番は書かれた順のまま）
    args.subset = list(dict.fromkeys(args.subset))
    return args


def default_name(args):
    """--name を省略したときの名前（bbox.json の入ったフォルダ名、または設定ファイル名）"""
    if args.pred is not None:
        return os.path.basename(os.path.dirname(os.path.abspath(args.pred)))
    name = os.path.splitext(os.path.basename(args.config))[0]
    # 変形後の座標の図は、元座標の図と同じ名前で上書きしないよう印を付ける
    return name + "_deformed" if args.coords == "deformed" else name


def add_project_path():
    """
    新手法の設定は custom_imports で edge_poisson_asap を読み込む（設定を読むだけで読み込まれる）ので、
    PYTHONPATH を設定しなくても読めるよう projects/Edge_poisson_asap を探索パスに加える
    """
    project_dir = os.path.dirname(TOOLS_DIR)
    if project_dir not in sys.path:
        # tools/ の直後に入れる（tools/ のモジュールが、同じ名前のほかのファイルより先に読まれるように）
        sys.path.insert(sys.path.index(TOOLS_DIR) + 1, project_dir)


def run_inference(config_path, checkpoint_path, out_dir, score_thr=None):
    """
    学習重みで val（設定の test_dataloader）の推論を行い、out_dir/results.bbox.json を作ってそのパスを返す。
    mmdet の tools/test.py と同じく Runner.test() を使い、test_evaluator.outfile_prefix だけ書き換える。
    非一様縮小の設定では評価クラスが元座標に戻すので、元座標の結果になる。
    score_thr: None でなければ model.test_cfg.score_thr をその値にする（--cfg-options と同じ）
    """
    add_project_path()
    # mmengine は重いので、学習重みから推論するときだけ読み込む
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(config_path)
    cfg.load_from = checkpoint_path
    if score_thr is not None:
        cfg.model.test_cfg.score_thr = score_thr
    cfg.work_dir = os.path.join(out_dir, "test_work_dir")    # ログの置き場所
    if isinstance(cfg.test_evaluator, (list, tuple)):
        raise ValueError("test_evaluator が複数ある設定には対応していません（1つにしてください）")
    # 評価クラスに検出結果を json で書き出させる（out_dir/results.bbox.json になる）
    prefix = os.path.join(os.path.abspath(out_dir), "results")
    cfg.test_evaluator.outfile_prefix = prefix
    Runner.from_cfg(cfg).test()
    return prefix + ".bbox.json"


def run_inference_deformed(config_path, checkpoint_path, gt_path, warp_dir, out_dir,
                           device="cuda:0", score_thr=None):
    """
    学習重みで、変形後の画像（warp_dir/images）をそのまま推論し、変形後の座標の検出結果を
    out_dir/pred_deformed.json に保存してそのパスを返す。
    推論する画像とカテゴリの対応は、変形後の座標の正解 gt_path（images と categories）から取る。
    推論は deformed_predict.run_detection（eval_deformed_orig_sizes.py と同じ）。元の座標へは戻さない。
    """
    add_project_path()
    # mmdet は重いので、変形後の座標で推論するときだけ読み込む
    from deformed_predict import label_to_cat_from_model, load_model, run_detection, save_preds
    from eval_common import load_json

    gt = load_json(gt_path)
    model = load_model(config_path, checkpoint_path, device, score_thr)
    label_to_cat = label_to_cat_from_model(model, gt["categories"])
    images = sorted(gt["images"], key=lambda im: im["id"])
    _, det_def, _ = run_detection(model, images, warp_dir, label_to_cat)   # box_method なし = 変形後のまま
    save_preds(out_dir, det_def)
    return os.path.join(out_dir, "pred_deformed.json")


def print_stats(name, st):
    """割り当ての集計を画面に表示する（score_thr の確認や件数の把握用）"""
    print(f"[{name}] 画像 {st['n_images']} 枚, 検出 {st['n_det']} 個, "
          f"重複として省いた検出 {st['n_dup']} 個, 残した検出 {st['n_kept']} 個"
          f"（うちクラス一致 {st['n_match']} 個）, "
          f"1枚あたりの最大検出数 {st['max_per_img']}, 最小スコア {st['min_score']:.4f}")


def _fmt(v):
    """表の値を文字にする（None は「-」、小数は3桁）"""
    if v is None:
        return "-"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def _pad(text, width):
    """全角文字を2文字分として数え、表示幅が width になるよう右側を空白で埋める（表の列をそろえるため）。
    幅の数え方は eval_common.text_width（mislocalized_cases の表と共通）"""
    return text + " " * max(width - text_width(text), 0)


def write_summary(blocks, name, path):
    """
    件数表を区切りごとに画面に表示し、まとめて1つの CSV に保存する
    （以前の write_summary を1手法用にしたもの。CSV の列は score_filter, size, item, 手法名）
    """
    width = max(16, len(name) + 2)     # 値の列の幅（手法名が長くても列がくっつかないように）
    for s_name, size_name, rows in blocks:
        print(f"\n=== 件数表: {s_name} / サイズ {size_name} ===")
        print(_pad("項目", 44) + f"{name:>{width}}")
        for label, value in rows:
            print(_pad(label, 44) + f"{_fmt(value):>{width}}")

    # Excel で開いても文字化けしないよう、BOM 付きの UTF-8 で保存する
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["score_filter", "size", "item", name])
        for s_name, size_name, rows in blocks:
            for label, value in rows:
                w.writerow([s_name, size_name, label.strip(), _fmt(value)])


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
    if args.pred is not None:
        pred_path = args.pred
    elif args.coords == "deformed":
        pred_path = run_inference_deformed(args.config, args.checkpoint, args.gt, args.warp_dir,
                                           args.out_dir, args.device, args.score_thr)
    else:
        pred_path = run_inference(args.config, args.checkpoint, args.out_dir, args.score_thr)
    name = args.name or default_name(args)

    # 集計: 検出と正解を対応付ける（重複の除去まで。subset での絞り込みは図ごとに行う）
    gt = load_gt(args.gt, size_gt_path=args.size_gt)
    matched, stats = match_all(gt, load_dets(pred_path))
    print_stats(name, stats)

    # 描画: 指定された subset ごとに3つの図を保存する
    prefixes = [save_figures(matched, subset, name, args) for subset in args.subset]

    # 件数表: 重複を省いた全検出から、スコアの絞り込み × サイズ ごとに作る
    blocks = build_summaries(matched, stats["raw_scores"], args.iou_thr, args.summary_score_thrs)
    summary_path = os.path.join(args.out_dir, f"{name}_summary.csv")
    write_summary(blocks, name, summary_path)

    print("\n注記: my_val.json には VisDrone の無視領域（ignored regions / others）が入っていないため、"
          "無視領域への検出も相手なし（IoU = 0）として数えている")
    for prefix in prefixes:
        print(f"保存先: {prefix}_{{scatter, hist_iou, hist_score}}.png")
    print(f"保存先: {summary_path}")


if __name__ == "__main__":
    main()