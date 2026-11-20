"""
検出器の推論と、検出結果ファイルの読み込みをまとめたモジュール（ツールどうしで共通に使う）

使っているツール:
  eval_deformed_orig_sizes.py : 変形後の座標で AP を測る（run_detection, load_pred_deformed など）
  draw_detection.py           : 1枚ずつ推論して正解と検出を描く（detect, load_inverse_map など）
  draw_mislocalized.py        : 推論して位置ずれの検出を描く（run_detection, print_settings など）

以前は各ツールに同じ処理がそれぞれ書かれていたものを、ここに1つにまとめた:
  load_inverse_map      ← draw_mislocalized.load_inverse_map（map_x がなければ止める版）
  detect                ← draw_detection.detect / draw_mislocalized.detect
  to_det_dicts          ← draw_mislocalized.to_det_dicts（= 以前の deformed_predict.to_results）
  run_detection         ← draw_mislocalized.run_detection（= 以前の deformed_predict.predict_dataset）
  label_to_cat_from_model ← 各ツールの main にあった「クラス名で対応させる」部分
  print_settings        ← draw_mislocalized.print_settings / draw_detection.prepare_warped の設定表示

推論の流れ:
  変形後の画像（<warp-dir>/images）をそのまま検出器に通すと、変形後の座標の検出結果が出る。
  同じ検出を、保存済みの逆写像（<warp-dir>/meshes/*.npz）と inverse_bbox で元座標に戻せる
  （OriginalCoordsCocoMetric と同じ方式）。
  一様縮小の設定では元画像を渡せば、mmdet が元画像の座標に戻して返す。

置き場所: projects/Edge_poisson_asap/tools/deformed_predict.py
mmdet と edge_poisson_asap は、使う関数の中でだけ読み込む
（ファイルの読み込みだけなら、mmdet や PYTHONPATH がなくても使えるようにするため）。
"""
import json
import os
import pickle

import numpy as np

from eval_common import load_json


# ===========================================================================
# 検出器の準備
# ===========================================================================
def load_model(config, checkpoint, device="cuda:0", score_thr=None):
    """
    検出器を読み込む。
    score_thr を渡したときは model.test_cfg.score_thr をその値にする
    （test を実行したときの --cfg-options model.test_cfg.score_thr=... と同じ値にして、
      AP の条件をそろえるため。tools/test.py 自体に決まった値があるわけではない）。
    None なら設定ファイルの値のまま（描画のツールはこちら）。
    """
    from mmengine.config import Config
    from mmdet.apis import init_detector
    cfg = Config.fromfile(config)
    if score_thr is not None:
        cfg.model.test_cfg.score_thr = score_thr
    return init_detector(cfg, checkpoint, device=device)


def label_to_cat_from_model(model, categories):
    """
    検出器のクラス番号 → 正解 json のカテゴリ id の対応（リスト）を作る。
    クラス名で対応させるので、設定の classes と json の categories の並び順が違っても合う。
    categories: 正解 json の categories
    """
    name_to_cat = {c["name"]: c["id"] for c in categories}
    return [name_to_cat[n] for n in model.dataset_meta["classes"]]


def print_settings(warp_dir, box_mode=None, n_per_side=None):
    """変形後のデータの作成時の設定（settings.json）を表示し、逆変換の方式を決めて返す。
    box_mode, n_per_side: 引数で指定された方式（None なら settings.json の値を使う）
    Returns: (box_mode, n_per_side)  実際に使う方式"""
    from edge_poisson_asap.box_method import resolve_box_method
    path = os.path.join(warp_dir, "settings.json")
    if os.path.isfile(path):
        print(f"変形後のデータの作成時の設定: {load_json(path)}")
    # 指定 > settings.json > 既定値 の順で決め、正解を作ったときと違えば警告する
    return resolve_box_method(warp_dir, box_mode, n_per_side)


def load_inverse_map(warp_dir, stem):
    """make_deformed_dataset.py --save-mesh で保存した逆写像 (map_x, map_y) を読む。
    npz がなければ None。npz はあるが逆写像が入っていなければ、作り直しが必要なので止める"""
    path = os.path.join(warp_dir, "meshes", stem + ".npz")
    if not os.path.isfile(path):
        return None
    with np.load(path) as z:
        if "map_x" not in z.files:
            raise KeyError(f"{path} に map_x がありません（中身: {z.files}）。"
                           "逆写像を保存する版の make_deformed_dataset.py --save-mesh でデータを作り直してください")
        return z["map_x"], z["map_y"]


# ===========================================================================
# 推論
# ===========================================================================
def detect(model, img, label_to_cat, score_thr=None):
    """
    画像1枚で推論し、検出を返す。
    img: 画像のパスか、画像の配列（BGR）。
         パスなら設定の test パイプラインの読み込み処理（LoadImageFromFile）を通る（tools/test.py と同じ）。
         配列なら読み込み処理が配列用になる。
    非一様縮小では変形後の画像を渡すので、変形後の座標で返る。
    一様縮小では元画像を渡すので、mmdet が元画像の座標に戻して返す。
    score_thr: None なら絞り込まない（model.test_cfg.score_thr に任せる）。数値ならそれ以上の検出だけ返す
    Returns: ボックス (K, 4) [x, y, w, h], スコア (K,), カテゴリ id のリスト (K,)
    """
    from mmdet.apis import inference_detector
    inst = inference_detector(model, img).pred_instances
    keep = np.ones(len(inst.scores), dtype=bool) if score_thr is None \
        else (inst.scores >= score_thr).cpu().numpy()          # スコアが下限以上の検出
    xyxy = inst.bboxes.cpu().numpy()[keep]
    xywh = np.c_[xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]]       # x1y1x2y2 -> COCO の xywh（+1 しない）
    scores = inst.scores.cpu().numpy()[keep]
    cats = [label_to_cat[int(l)] for l in inst.labels.cpu().numpy()[keep]]
    return xywh, scores, cats


def to_det_dicts(image_id, boxes, scores, cats):
    """検出を COCO の結果形式（results.bbox.json と同じ形）の dict のリストにする"""
    return [dict(image_id=int(image_id), category_id=int(c), bbox=[float(v) for v in b], score=float(s))
            for b, s, c in zip(boxes, scores, cats)]


def run_detection(model, images, warp_dir, label_to_cat, box_method=None, score_thr=None,
                  skip_missing=False):
    """
    変形後の画像をまとめて推論する。
    images      : 推論する変形後の画像の情報（変形後の正解 json の images の要素。id と file_name を使う）。
                  画像 id は元のデータと同じ（make_deformed_dataset.py は id を変えない）
    box_method  : (bbox_mode, n_per_side)。渡したときは、同じ検出を逆写像で元の座標へ戻したものも作る。
                  None なら元の座標へは戻さない（逆写像も読まない）
    score_thr   : detect と同じ（None なら絞り込まない）
    skip_missing: 逆写像の npz がない画像を、True なら飛ばす（skipped に名前を入れる）、False なら止める
    Returns: (元の座標の検出 または None, 変形後の座標の検出, 飛ばした画像の名前のリスト)
             検出の2つは results.bbox.json と同じ形で、同じ並び
             （同じ並びにしておくと、あとで2つを順番で1対1に対応付けられる）
    """
    if box_method is not None:
        from edge_poisson_asap.box_transform import inverse_bbox   # 元の座標へ戻すときだけ使う
    det_orig = [] if box_method is not None else None
    det_def, skipped = [], []
    for n, im in enumerate(images, 1):
        maps = None
        if box_method is not None:
            maps = load_inverse_map(warp_dir, os.path.splitext(im["file_name"])[0])
            if maps is None:
                if not skip_missing:
                    raise FileNotFoundError(f"逆写像がありません（--save-mesh なしで作ったデータ？）: {im['file_name']}")
                print(f"[スキップ] 逆写像の npz がありません: {im['file_name']}")
                skipped.append(im["file_name"])
                continue
        boxes, scores, cats = detect(model, os.path.join(warp_dir, "images", im["file_name"]),
                                     label_to_cat, score_thr)
        det_def += to_det_dicts(im["id"], boxes, scores, cats)
        if maps is not None:
            # 同じ検出を、保存済みの逆写像で元の座標に戻す（検出が0個の画像は戻す処理を呼ばない）
            orig_boxes = inverse_bbox(boxes, *maps, *box_method) if len(boxes) else boxes
            det_orig += to_det_dicts(im["id"], orig_boxes, scores, cats)
        if n % 50 == 0 or n == len(images):
            print(f"推論 {n}/{len(images)} 枚")
    return det_orig, det_def, skipped


# ===========================================================================
# 検出結果の保存と読み込み
# ===========================================================================
def save_preds(save_dir, preds_deformed, preds_orig=None):
    """推論した検出結果を保存する（次からは読み込めば推論しなくて済む）"""
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "pred_deformed.json"), "w") as f:
        json.dump(preds_deformed, f)
    if preds_orig is not None:
        with open(os.path.join(save_dir, "pred_orig.json"), "w") as f:
            json.dump(preds_orig, f)
    print(f"検出結果を保存しました: {save_dir}")


def _to_numpy(x):
    """torch.Tensor でも numpy でもリストでも numpy 配列にする（pkl の中身は CPU の Tensor）"""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def make_label2cat(categories, classes=None):
    """
    pkl 用（検出器がないとき）: 検出器のラベル番号（0, 1, ...）→ category_id の対応を作る。
    mmdet の CocoDataset と同じく、正解 json の categories の並び順で、
    classes（設定の metainfo の classes）に含まれるものだけを順に 0, 1, ... とする。
    """
    cats = categories if classes is None else [c for c in categories if c["name"] in classes]
    return {label: c["id"] for label, c in enumerate(cats)}


def load_pred_deformed(path, label2cat=None):
    """
    検出結果を COCO の結果形式で読む（名前は変形後の座標用だが、どの座標の json でも読める）。
      .json: そのまま返す（results.bbox.json、save_preds の pred_deformed.json など）
      .pkl : tools/test.py --out の出力。CocoMetric の results2json と同じ順番に並べる
    """
    if path.endswith(".json"):
        return load_json(path)

    with open(path, "rb") as f:
        samples = pickle.load(f)                       # torch がインストールされた環境で読む必要がある
    preds = []
    for s in samples:
        inst = s["pred_instances"]
        bboxes = _to_numpy(inst["bboxes"]).reshape(-1, 4)
        xywh = np.c_[bboxes[:, :2], bboxes[:, 2:] - bboxes[:, :2]]
        cats = [label2cat[int(l)] for l in _to_numpy(inst["labels"]).reshape(-1)]
        preds += to_det_dicts(s["img_id"], xywh, _to_numpy(inst["scores"]).reshape(-1), cats)
    return preds