"""位置ずれの検出（クラスは正しいが IoU が低い検出）を集めるモジュール。

割り当ての規則は plot_score_iou.py と同じにしている：
  - クラスを無視して、IoU が最大の正解を相手にする
  - 同じ正解を指す検出が複数あれば、スコア最大の1つだけ残す（残りは重複として除く）
  - IoU=0 の検出は相手なし
"""
import json
import os
from collections import defaultdict

import numpy as np


def load_json(path):
    """json ファイルを読み込んで返す"""
    with open(path) as f:
        return json.load(f)


def size_label(area):
    """元画像での面積から COCO のサイズ区分を返す（境界は 32², 96²）"""
    if area < 32 ** 2:
        return 'small'
    if area < 96 ** 2:
        return 'medium'
    return 'large'


def iou_matrix(dets_xywh, gts_xywh):
    """検出 (N,4) と正解 (M,4) の IoU 行列 (N,M) を返す。ボックスは COCO 形式 [x, y, w, h]"""
    d = np.asarray(dets_xywh, dtype=np.float64).reshape(-1, 4)
    g = np.asarray(gts_xywh, dtype=np.float64).reshape(-1, 4)
    # 左上・右下の座標に直す（検出は列ベクトル、正解は行ベクトルにして総当たりで計算）
    dx1, dy1 = d[:, 0:1], d[:, 1:2]
    dx2, dy2 = dx1 + d[:, 2:3], dy1 + d[:, 3:4]
    gx1, gy1 = g[:, 0], g[:, 1]
    gx2, gy2 = gx1 + g[:, 2], gy1 + g[:, 3]
    # 重なり部分の幅と高さ（重ならなければ 0）
    iw = np.clip(np.minimum(dx2, gx2) - np.maximum(dx1, gx1), 0, None)
    ih = np.clip(np.minimum(dy2, gy2) - np.maximum(dy1, gy1), 0, None)
    inter = iw * ih
    union = d[:, 2:3] * d[:, 3:4] + g[:, 2] * g[:, 3] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def index_by_image(items):
    """検出や正解を image_id ごとのリストにまとめる"""
    out = defaultdict(list)
    for it in items:
        out[it['image_id']].append(it)
    return out


def map_image_ids(gt_orig, gt_def):
    """元画像の image_id → 非一様縮小後の image_id の対応を、ファイル名（拡張子を除く）で作る"""
    def stem(name):
        # ディレクトリと拡張子を除いたファイル名
        return os.path.splitext(os.path.basename(name))[0]
    def_by_stem = {stem(im['file_name']): im['id'] for im in gt_def['images']}
    return {im['id']: def_by_stem.get(stem(im['file_name'])) for im in gt_orig['images']}


def match_one_image(dets, gts):
    """1枚分の検出を正解に割り当てる。
    戻り値: {検出の添字: (正解の添字, IoU)}。相手なし（IoU=0）と重複は含まない"""
    if not dets or not gts:
        return {}
    ious = iou_matrix([d['bbox'] for d in dets], [g['bbox'] for g in gts])
    best_gt = ious.argmax(axis=1)                      # クラスを無視して IoU 最大の正解
    best_iou = ious[np.arange(len(dets)), best_gt]
    keep = {}                                          # 正解の添字 → 残す検出の添字
    for i in range(len(dets)):
        if best_iou[i] <= 0:
            continue                                   # IoU=0 は相手なし
        j = int(best_gt[i])
        # 同じ正解を指す検出のうち、スコアが最大のものだけ残す
        if j not in keep or dets[i]['score'] > dets[keep[j]]['score']:
            keep[j] = i
    return {i: (j, float(best_iou[i])) for j, i in keep.items()}


def pair_detections(dets_o, dets_d):
    """同じ画像の「元座標の検出」と「非一様縮小後の座標の検出」を1対1に対応付ける。
    同じ推論結果を座標だけ変えて保存したものなので、まず並び順で対応付ける。
    並び順でクラスかスコアが食い違ったら、(クラス, スコア) が同じものどうしで対応付ける。
    戻り値: ({元座標の検出の添字: 非一様縮小後の検出 or None}, 並び順で対応付けられたか)"""
    same_order = len(dets_o) == len(dets_d) and all(
        a['category_id'] == b['category_id'] and abs(a['score'] - b['score']) < 1e-6
        for a, b in zip(dets_o, dets_d))
    if same_order:
        return dict(enumerate(dets_d)), True
    # 予備の方法：(クラス, スコア) をキーにして対応付ける（同じスコアが並ぶと取り違える可能性あり）
    pool = defaultdict(list)
    for d in dets_d:
        pool[(d['category_id'], round(d['score'], 5))].append(d)
    pairs = {}
    for i, d in enumerate(dets_o):
        cands = pool.get((d['category_id'], round(d['score'], 5)))
        pairs[i] = cands.pop(0) if cands else None
    return pairs, False


def collect_cases(gt_orig, gt_def, det_orig, det_def, score_thr=0.01, iou_min=0.0, iou_max=0.5):
    """クラス正解かつ iou_min <= IoU < iou_max（元座標）の検出を集め、
    同じ正解に対する非一様縮小後の座標での IoU も添える。
    戻り値: (検出ごとの dict のリスト, 並び順で対応付けられなかった画像の数)"""
    img_map = map_image_ids(gt_orig, gt_def)
    gts_o = index_by_image(gt_orig['annotations'])
    gt_def_by_id = {a['id']: a for a in gt_def['annotations']}   # 変形後の正解は annotation id で対応付ける
    # 表1・2と同じく、スコアで絞ってから割り当てる（重複の除き方が閾値で変わるため）
    dets_o_all = index_by_image(d for d in det_orig if d['score'] >= score_thr)
    dets_d_all = index_by_image(d for d in det_def if d['score'] >= score_thr)

    cases, n_fallback = [], 0
    for img_id, dets in dets_o_all.items():
        gts = gts_o.get(img_id, [])
        img_id_def = img_map.get(img_id)
        pairs, ordered = pair_detections(dets, dets_d_all.get(img_id_def, []))
        n_fallback += int(not ordered)
        for i, (j, iou_o) in match_one_image(dets, gts).items():
            det, gt = dets[i], gts[j]
            if det['category_id'] != gt['category_id'] or not (iou_min <= iou_o < iou_max):
                continue                               # クラス違い、または IoU が対象の範囲外
            det_d, gt_d = pairs.get(i), gt_def_by_id.get(gt['id'])
            # 非一様縮小後の座標では、割り当て直さず「同じ正解」との IoU を測る
            iou_d = (float(iou_matrix([det_d['bbox']], [gt_d['bbox']])[0, 0])
                     if det_d is not None and gt_d is not None else float('nan'))
            area = gt.get('area', gt['bbox'][2] * gt['bbox'][3])
            cases.append(dict(
                image_id=img_id, image_id_def=img_id_def, ann_id=gt['id'],
                category_id=gt['category_id'], size=size_label(area), score=det['score'],
                iou_orig=iou_o, iou_def=iou_d,
                gt_bbox_orig=gt['bbox'], det_bbox_orig=det['bbox'],
                gt_bbox_def=gt_d['bbox'] if gt_d is not None else None,
                det_bbox_def=det_d['bbox'] if det_d is not None else None))
    return cases, n_fallback


if __name__ == '__main__':
    # 動作確認：小さな作り物のデータで割り当てと IoU を確かめる
    gts = [dict(id=1, bbox=[0, 0, 10, 10], category_id=1, area=100),
           dict(id=2, bbox=[50, 50, 10, 10], category_id=2, area=100)]
    dets = [dict(bbox=[2, 0, 10, 10], score=0.9, category_id=1),    # 正解1と IoU 0.667
            dict(bbox=[5, 0, 10, 10], score=0.5, category_id=1),    # 正解1の重複（スコアが低いので除かれる）
            dict(bbox=[55, 55, 10, 10], score=0.8, category_id=2),  # 正解2と IoU 0.143
            dict(bbox=[200, 200, 5, 5], score=0.7, category_id=1)]  # 相手なし
    print(match_one_image(dets, gts))  # 期待値: {0: (0, 0.667), 2: (1, 0.143)}