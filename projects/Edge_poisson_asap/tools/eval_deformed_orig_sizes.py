"""
非一様縮小後の座標（検出器の入力座標）で AP を測る評価スクリプト
（変形後の正解 × 変形後の検出結果。サイズ区分は元画像の面積か、変形後の面積かを選べる）

切り分け用: 「元座標に戻す処理」を通さずに検出器そのものの AP を見る。
  - IoU は非一様縮小後の座標の正解（<warp-dir>/annotations.json）と
    非一様縮小後の座標の検出結果で測る
  - 非一様縮小後の座標の検出結果は、次のどちらかで手に入れる（deformed_predict.py）
      推論する : --config と --checkpoint を渡すと、このスクリプトの中で変形後の画像を推論する
                 （設定の評価クラスが OriginalCoordsCocoMetric になり、tools/test.py の
                   results.bbox.json は元座標のものしか出ないため）
      読み込む : --pred-deformed に .json か .pkl を渡す
                 （推論したときに --save-pred-dir に保存した pred_deformed.json、tools/test.py --out の pkl など）
  - サイズ区分（small / medium / large）は --area-mode で選ぶ
      orig     : 元画像の面積（一様縮小とサイズ区分をそろえて比べたいとき。既定）
                   正解     : 元の正解（my_val.json）の area をアノテーション id でつないで置き換える
                   検出結果 : 元座標に戻した箱の面積を使う。どの正解にも当たらなかった検出は、
                              COCO の評価では検出自身の面積で「その区分の評価に入れるか」を決めるため、これもそろえる
                              推論したとき → 逆写像で元座標に戻した箱（OriginalCoordsCocoMetric と同じ方式）
                              読み込むとき → --pred-orig の json（同じ推論結果を同じ順で書き出したもの）
      deformed : 変形後の面積のまま（正解・検出とも変形後の画像だけで閉じた評価）
  - 最大検出数は mmdet の CocoMetric と同じ 100, 300, 1000（AP は 100 個まで使った値）
  - 全体の AP のあとに、クラス別の AP（mAP, mAP_50, mAP_75, mAP_s, mAP_m, mAP_l）の表も出す
    （mmdet の CocoMetric の classwise=True と同じ計算。そのクラスの正解も検出もない区分は nan）

置き場所: projects/Edge_poisson_asap/tools/eval_deformed_orig_sizes.py（deformed_predict.py, eval_common.py と同じフォルダ）

使い方（workspace/ で実行）:
  画面に出る内容（設定の表示・推論の進み具合・COCO の表・クラス別の表）は、tee で .log にも保存する
  （2>&1 でエラーも同じ .log に残す。tee の保存先のフォルダは先に作っておく）

  # 推論して評価する（サイズ区分は元画像の面積）。検出結果は保存しておき、次からは読み込みで済ませる
  mkdir -p $OUT/deformed_eval
  PYTHONPATH=projects/Edge_poisson_asap:$PYTHONPATH python projects/Edge_poisson_asap/tools/eval_deformed_orig_sizes.py \
    --warp-dir data_ann/VisDrone_poisson_asap_mean3_3-05/val \
    --gt-orig data_ann/VisDrone/my_val.json \
    --config projects/Edge_poisson_asap/configs/yolox_tiny_visdrone_poisson_asap_mean3_3-05.py \
    --checkpoint work_dirs/yolox_tiny_visdrone_poisson_asap_mean3_3-05/epoch_50.pth \
    --score-thr 0.01 --save-pred-dir work_dirs/eval_visdrone_poisson_asap_mean3_3-05/deformed_eval\
    2>&1 | tee work_dirs/eval_visdrone_poisson_asap_mean3_3-05/deformed_eval/eval_orig_sizes.log

  # 保存した検出結果で、サイズ区分を変形後の面積にして評価し直す（推論しない）
  # （引数の定義に edge_poisson_asap.box_method を使うので、読み込みのときも PYTHONPATH は必要）
  PYTHONPATH=projects/Edge_poisson_asap:$PYTHONPATH python projects/Edge_poisson_asap/tools/eval_deformed_orig_sizes.py \
      --warp-dir data_ann/VisDrone_poisson_asap/val \
      --pred-deformed $OUT/deformed_eval/pred_deformed.json \
      --area-mode deformed \
      2>&1 | tee $OUT/deformed_eval/eval_deformed_sizes.log
"""
import argparse
import os

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from deformed_predict import (label_to_cat_from_model, load_model, load_pred_deformed, make_label2cat,
                              print_settings, run_detection, save_preds)
from eval_common import load_json, replace_area_by_id, text_width

# クラス別の表の列: (列名, IoU の閾値の番号（None = 0.50:0.95 の平均）, 面積の区分の番号)
# COCOeval の precision の並び: IoU 0.50, 0.55, ..., 0.95 / 面積 all, small, medium, large
CLASSWISE_COLUMNS = [("mAP", None, 0), ("mAP_50", 0, 0), ("mAP_75", 5, 0),
                     ("mAP_s", None, 1), ("mAP_m", None, 2), ("mAP_l", None, 3)]


# ===========================================================================
# 正解の読み込み
# ===========================================================================
def load_gt_with_orig_area(gt_deformed_path, gt_orig_path=None):
    """
    非一様縮小後の座標の正解を読んで COCO を作る。
    gt_orig_path を渡したときは、area だけ元の正解のものに置き換える（サイズ区分を元画像の面積にする）。
    gt_orig_path が None なら、area は変形後の正解のまま。
    """
    deformed = load_json(gt_deformed_path)
    if gt_orig_path is not None:
        # 箱は非一様縮小後の座標のまま、面積だけ元画像のもの（eval_common と score_iou_match で共通の処理）
        deformed["annotations"] = replace_area_by_id(deformed["annotations"], gt_orig_path)
    coco = COCO()
    coco.dataset = deformed
    coco.createIndex()
    return coco


# ===========================================================================
# 検出結果の元画像での面積（--area-mode orig のときだけ使う）
# ===========================================================================
def orig_det_areas(pred_deformed, pred_orig):
    """
    2つの検出結果を順番で対応させ、元座標に戻した箱の面積を返す。
    対応が崩れていないか（画像 id・クラス・スコアが同じか）を確認する。
    さらに、2つの箱がすべて同じでないか（変形後のつもりが元座標になっていないか）も確認する。
    """
    if len(pred_deformed) != len(pred_orig):
        raise ValueError(f"検出数が違います: {len(pred_deformed)} と {len(pred_orig)}")
    for i, (d, o) in enumerate(zip(pred_deformed, pred_orig)):
        if (d["image_id"] != o["image_id"] or d["category_id"] != o["category_id"]
                or abs(d["score"] - o["score"]) > 1e-6):
            raise ValueError(f"{i} 番目の検出が対応していません: {d} / {o}")
    if pred_deformed:
        b_def = np.array([d["bbox"] for d in pred_deformed])
        b_orig = np.array([o["bbox"] for o in pred_orig])
        if np.allclose(b_def, b_orig, atol=1e-3):
            raise ValueError("変形後の検出結果と元座標の検出結果の箱がすべて同じです。"
                             "変形後のつもりの検出結果が元座標になっている可能性があります")
    return [o["bbox"][2] * o["bbox"][3] for o in pred_orig]


# ===========================================================================
# COCO 評価
# ===========================================================================
def evaluate(coco_gt, pred_deformed, det_areas=None):
    """
    COCO 評価する（最大検出数は mmdet と同じ）。
    det_areas を渡したときは、検出の area を元画像の面積に置き換えてから評価する。
    None なら、検出の area は変形後の箱の大きさのまま。
    """
    coco_dt = coco_gt.loadRes(pred_deformed)
    if det_areas is not None:
        # loadRes は渡した順に id を 1, 2, ... と付け、area を箱の大きさ（非一様縮小後）で計算する。
        # その area を元画像の面積に置き換える
        for ann_id, area in zip(sorted(coco_dt.anns), det_areas):
            coco_dt.anns[ann_id]["area"] = area
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.maxDets = [100, 300, 1000]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats
    metrics = {"mAP": s[0], "mAP_50": s[1], "mAP_75": s[2], "mAP_s": s[3], "mAP_m": s[4], "mAP_l": s[5]}
    return metrics, classwise_ap(ev, coco_gt)


def classwise_ap(ev, coco_gt):
    """
    クラス別の AP を求める（mmdet の CocoMetric の classwise=True と同じ計算）。
    ev.eval["precision"] は [IoU の閾値, 再現率, クラス, 面積の区分, 最大検出数] の配列で、
    計算できない所は -1 が入っている。-1 を除いて平均したものが AP（全部 -1 なら nan）。
    最大検出数は最後（1000 個）を使う（全体の mAP と同じ）。
    Returns: [(クラス名, {列名: AP}), ...]（クラスは COCOeval が使った category_id の順）
    """
    precisions = ev.eval["precision"]
    rows = []
    for k, cat_id in enumerate(ev.params.catIds):
        # loadCats は id が numpy の整数だと None を返すことがあるので、辞書から直接引く
        name = coco_gt.cats[int(cat_id)]["name"]
        values = {}
        for col, iou_idx, area_idx in CLASSWISE_COLUMNS:
            p = precisions[:, :, k, area_idx, -1] if iou_idx is None else precisions[iou_idx, :, k, area_idx, -1]
            p = p[p > -1]                                      # 計算できない所（-1）は除く
            values[col] = float(np.mean(p)) if p.size else float("nan")
        rows.append((name, values))
    return rows


def print_classwise(rows):
    """クラス別の AP を表にして表示する（クラス名が日本語でも列がそろうよう、表示幅で空白を足す）"""
    name_w = max([text_width("class")] + [text_width(n) for n, _ in rows])
    print("\nクラス別の AP:")
    print("class" + " " * (name_w - text_width("class")) + "".join(f"{c:>9}" for c, _, _ in CLASSWISE_COLUMNS))
    for name, values in rows:
        print(name + " " * (name_w - text_width(name))
              + "".join(f"{values[c]:>9.3f}" for c, _, _ in CLASSWISE_COLUMNS))


# ===========================================================================
# 引数と全体の流れ
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="非一様縮小後の座標で AP を測る（サイズ区分は元画像か変形後の面積）")
    p.add_argument("--warp-dir", required=True,
                   help="make_deformed_dataset.py の --out-dir（images/, annotations.json, meshes/ がある所）")
    p.add_argument("--gt-deformed", default=None,
                   help="非一様縮小後の座標の正解（省略すると <warp-dir>/annotations.json）")
    p.add_argument("--area-mode", choices=["orig", "deformed"], default="orig",
                   help="サイズ区分に使う面積。orig=元画像（--gt-orig が必要）、deformed=変形後のまま")
    p.add_argument("--gt-orig", default=None, help="元画像の座標の正解（--area-mode orig で正解の面積に使う）")
    # 検出結果を推論で作るとき
    p.add_argument("--config", default=None, help="mmdet の設定ファイル（推論するとき）")
    p.add_argument("--checkpoint", default=None, help="学習済みの重み（推論するとき）")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--score-thr", type=float, default=None,
                   help="推論するときの model.test_cfg.score_thr（省略時は設定ファイルの値）。"
                        "test のときに --cfg-options model.test_cfg.score_thr=0.01 を付けていたなら 0.01 を指定")
    p.add_argument("--save-pred-dir", default=None,
                   help="推論した検出結果（pred_deformed.json / pred_orig.json）の保存先")
    # 検出結果を読み込むとき
    p.add_argument("--pred-deformed", default=None, help="変形後の座標の検出結果（.json か .pkl）")
    p.add_argument("--pred-orig", default=None,
                   help="元座標の検出結果（読み込むときの --area-mode orig で検出の面積に使う）")
    p.add_argument("--classes", nargs="*", default=None,
                   help="pkl を読むときだけ使う: 設定の metainfo の classes（ラベル番号→category_id の対応用）")
    # 推論した検出を元座標へ戻すときのボックスの作り方。省略すると --warp-dir の settings.json の値
    from edge_poisson_asap.box_method import add_box_args
    add_box_args(p, from_settings=True)
    args = p.parse_args()

    use_infer = args.config is not None or args.checkpoint is not None
    if use_infer == (args.pred_deformed is not None):
        p.error("推論（--config と --checkpoint）か読み込み（--pred-deformed）のどちらか一方を指定してください")
    if use_infer and (args.config is None or args.checkpoint is None):
        p.error("推論するときは --config と --checkpoint の両方が必要です")
    if args.area_mode == "orig":
        if args.gt_orig is None:
            p.error("--area-mode orig には --gt-orig が必要です")
        if not use_infer and args.pred_orig is None:
            p.error("読み込みで --area-mode orig にするときは --pred-orig が必要です")
    return args, use_infer


def get_predictions(args, use_infer, coco_gt):
    """変形後の座標の検出結果と、元座標の検出結果（--area-mode orig のときだけ。ほかは None）を返す"""
    use_orig_area = args.area_mode == "orig"
    if use_infer:
        # 作成時の設定を表示し、検出を元の座標へ戻す方式を決める（指定 > settings.json > 既定値）。
        # 正解を作ったときと同じ方式にそろう
        box_method = print_settings(args.warp_dir, args.bbox_mode, args.bbox_n_per_side)
        print(f"元座標へ戻すボックスの作り方: {box_method}")
        model = load_model(args.config, args.checkpoint, args.device, args.score_thr)
        label_to_cat = label_to_cat_from_model(model, coco_gt.dataset["categories"])
        images = sorted(coco_gt.dataset["images"], key=lambda im: im["id"])
        # 変形後の画像をすべて推論する（サイズ区分を元画像の面積にするときは、逆写像で元座標にも戻す）
        preds_orig, preds_deformed, _ = run_detection(
            model, images, args.warp_dir, label_to_cat, box_method if use_orig_area else None)
        if args.save_pred_dir:
            save_preds(args.save_pred_dir, preds_deformed, preds_orig)
        return preds_deformed, preds_orig

    label2cat = make_label2cat(coco_gt.dataset["categories"], args.classes)
    preds_deformed = load_pred_deformed(args.pred_deformed, label2cat)
    preds_orig = None
    if use_orig_area:
        preds_orig = load_json(args.pred_orig)
    return preds_deformed, preds_orig


def main():
    args, use_infer = parse_args()
    gt_deformed = args.gt_deformed or os.path.join(args.warp_dir, "annotations.json")
    use_orig_area = args.area_mode == "orig"

    coco_gt = load_gt_with_orig_area(gt_deformed, args.gt_orig if use_orig_area else None)
    preds_deformed, preds_orig = get_predictions(args, use_infer, coco_gt)
    det_areas = orig_det_areas(preds_deformed, preds_orig) if use_orig_area else None

    metrics, classwise = evaluate(coco_gt, preds_deformed, det_areas)
    area_name = "元画像" if use_orig_area else "変形後"
    print_classwise(classwise)
    print(f"\nbbox（非一様縮小後の座標・{area_name}のサイズ区分）: "
          + ", ".join(f"{k} {v:.3f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()