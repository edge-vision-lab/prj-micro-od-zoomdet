"""
評価・確認ツールで共通に使う小さな関数（検出器や mmdet は使わない。numpy だけ）

以前は各ツールに同じ処理がそれぞれ書かれていたものを、ここに1つにまとめた:
  load_json          ← mislocalized_cases.load_json
  group_by_image     ← draw_detection.group_by_image / mislocalized_cases.index_by_image
  size_label         ← mislocalized_cases.size_label / magnification_lib.coco_size_category
  size_category      ← score_iou_match.size_category（配列版。番号で返す）
  iou_matrix         ← mislocalized_cases.iou_matrix / score_iou_match.iou_matrix
  replace_area_by_id ← score_iou_match.load_gt と eval_deformed_orig_sizes.load_gt_with_orig_area の
                       「面積だけ元の正解のものにする」部分
  text_width         ← mislocalized_cases.text_width / plot_score_iou._pad の幅の数え方

置き場所: projects/Edge_poisson_asap/tools/eval_common.py
（tools/ のほかのファイルから import する。tools/ のスクリプトを直接実行すれば読み込める）
動作確認: python projects/Edge_poisson_asap/tools/eval_common.py
"""
import json
import unicodedata
from collections import defaultdict

import numpy as np

# COCO の物体サイズの境界（元画像での面積 [ピクセル数]）。これ未満が small / medium
SMALL_MAX_AREA = 32 ** 2
MEDIUM_MAX_AREA = 96 ** 2

# サイズ区分の番号（配列で扱うとき。相手の GT がいない検出は SIZE_NONE）
SIZE_NONE, SIZE_SMALL, SIZE_MEDIUM, SIZE_LARGE = -1, 0, 1, 2


# ===========================================================================
# json の読み込みとまとめ方
# ===========================================================================
def load_json(path):
    """json ファイルを読み込んで返す"""
    with open(path) as f:
        return json.load(f)


def group_by_image(items):
    """正解や検出（dict のリスト）を image_id ごとのリストにまとめる"""
    out = defaultdict(list)
    for it in items:
        out[it["image_id"]].append(it)
    return out


def replace_area_by_id(anns, area_json_path):
    """
    アノテーション（dict のリスト）の area だけを、別の正解 json の area に置き換えた新しいリストを返す。
    箱は anns の座標のまま。変形後の座標の正解で IoU を測りつつ、サイズ区分は元画像の面積にするときに使う。
    2つの json はアノテーション id で対応させる（make_deformed_dataset.py は id を変えない）。
    """
    area_by_id = {a["id"]: a["area"] for a in load_json(area_json_path)["annotations"]}
    missing = [a["id"] for a in anns if a["id"] not in area_by_id]
    if missing:
        raise KeyError(f"{area_json_path} にないアノテーション id が {len(missing)} 個あります"
                       f"（例: {missing[:5]}）")
    return [{**a, "area": area_by_id[a["id"]]} for a in anns]


# ===========================================================================
# サイズ区分と IoU
# ===========================================================================
def size_label(area):
    """面積1つから COCO のサイズ区分の名前（small / medium / large）を返す"""
    if area < SMALL_MAX_AREA:
        return "small"
    if area < MEDIUM_MAX_AREA:
        return "medium"
    return "large"


def size_category(areas):
    """面積の配列から COCO のサイズ区分の番号（SIZE_SMALL など）の配列を返す"""
    areas = np.asarray(areas)
    return np.where(areas < SMALL_MAX_AREA, SIZE_SMALL,
                    np.where(areas < MEDIUM_MAX_AREA, SIZE_MEDIUM, SIZE_LARGE))


def iou_matrix(dets_xywh, gts_xywh):
    """検出 (N,4) と正解 (M,4) の IoU 行列 (N,M) を返す（クラスは見ない）。ボックスは COCO 形式 [x, y, w, h]"""
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


# ===========================================================================
# 表示
# ===========================================================================
def text_width(s):
    """表示したときの幅（全角文字は 2、半角文字は 1 として数える）。表の列をそろえるために使う"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(s))


if __name__ == "__main__":
    # 動作確認: 小さな作り物のデータで各関数を確かめる
    import os
    import tempfile
    print(iou_matrix([[0, 0, 10, 10]], [[5, 0, 10, 10], [100, 100, 1, 1]]))   # 期待値: [[0.333, 0]]
    assert np.allclose(iou_matrix([[0, 0, 10, 10]], [[5, 0, 10, 10]]), 1 / 3)
    assert [size_label(a) for a in (100, 1024, 9216)] == ["small", "medium", "large"]   # 境界は上の区分
    assert size_category([100, 1024, 9216]).tolist() == [SIZE_SMALL, SIZE_MEDIUM, SIZE_LARGE]
    assert {k: len(v) for k, v in group_by_image([{"image_id": 1}, {"image_id": 1}, {"image_id": 2}]).items()} \
        == {1: 2, 2: 1}
    assert text_width("ab") == 2 and text_width("あい") == 4
    # 面積の置き換え: 箱はそのまま、area だけ別の json のものになる
    path = os.path.join(tempfile.mkdtemp(), "orig.json")
    with open(path, "w") as f:
        json.dump({"annotations": [{"id": 5, "area": 400.0}]}, f)
    out = replace_area_by_id([{"id": 5, "bbox": [0, 0, 10, 10], "area": 100.0}], path)
    assert out == [{"id": 5, "bbox": [0, 0, 10, 10], "area": 400.0}]
    print("OK")