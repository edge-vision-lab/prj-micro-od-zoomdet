"""TIDE（Bolya ら, ECCV 2020）で検出の誤りを種類ごとに分けるための関数。

誤りの種類:
  Cls  : 位置は合っているがクラス違い
  Loc  : クラスは合っているが位置ずれ（背景の閾値 <= IoU < 正解の閾値）
  Both : クラス違いかつ位置ずれ
  Dupe : 重複（同じ正解に2つ目以降の検出）
  Bkg  : 背景への誤検出（どの正解とも IoU < 背景の閾値）
  Miss : 見逃し
  FalsePos / FalseNeg : 誤検出すべて / 見逃しすべて（上の6種類と重なる）
各値は「他の条件はそのままで、その種類の誤りだけを直したら AP が何ポイント上がるか」。

置き場所: projects/Edge_poisson_asap/tools/tide_lib.py（run_tide.py から使う）
"""
import json
import os
import pickle
import warnings

import numpy as np


def patch_numpy_aliases():
    """tidecv は古い numpy の別名（np.float など）を使っているので、
    新しい numpy（1.24 以降）でも動くよう、別名を足しておく"""
    for name, typ in (('float', float), ('int', int), ('bool', bool)):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)   # 別名があるかを調べるときの警告を抑える
            exists = hasattr(np, name)
        if not exists:
            setattr(np, name, typ)


patch_numpy_aliases()
from tidecv import TIDE, datasets  # noqa: E402  （別名を足してから読み込む）

MAIN_ERRORS = ['Cls', 'Loc', 'Both', 'Dupe', 'Bkg', 'Miss']   # 主な誤りの種類（表の列の順）
SPECIAL_ERRORS = ['FalsePos', 'FalseNeg']                      # 誤検出全体・見逃し全体
COLS = ['AP'] + MAIN_ERRORS + SPECIAL_ERRORS                   # 出力する列の順


def prepare_gt_for_tide(gt_path, out_dir):
    """tidecv の COCO 読み込みは、annotations に iscrowd と segmentation が必要。
    足りなければ足したコピーを out_dir に作り、そのパスを返す（元の json は変えない）。
    segmentation はボックスの4隅をそのまま多角形にしたもので、bbox の評価では使われない"""
    with open(gt_path) as f:
        coco = json.load(f)
    anns = coco['annotations']
    if all('iscrowd' in a and 'segmentation' in a for a in anns):
        return gt_path
    for a in anns:
        a.setdefault('iscrowd', 0)
        if 'segmentation' not in a:
            x, y, w, h = a['bbox']
            # 左上 → 右上 → 右下 → 左下 の順に並べた多角形
            a['segmentation'] = [[x, y, x + w, y, x + w, y + h, x, y + h]]
    out_path = os.path.join(out_dir, 'gt_for_tide.json')
    with open(out_path, 'w') as f:
        json.dump(coco, f)
    return out_path


def label_to_category_id(gt_path):
    """mmdet の pkl のラベル番号（0 始まり）→ COCO の category_id の対応を返す。
    mmdet 3.x の CocoDataset と同じく、正解 json の categories の並び順で番号を振る"""
    with open(gt_path) as f:
        cats = json.load(f)['categories']
    return {i: c['id'] for i, c in enumerate(cats)}


def _to_numpy(x):
    """torch のテンソルでも numpy 配列でも numpy 配列にして返す"""
    return x.cpu().numpy() if hasattr(x, 'cpu') else np.asarray(x)


def pkl_to_coco_dets(pkl_path, label2cat):
    """test.py --out で保存した pkl（画像ごとの検出結果のリスト）を、
    COCO の検出結果の形式（*.bbox.json と同じ dict のリスト）に直す。
    pkl のボックスは [x1, y1, x2, y2] なので、COCO の [x, y, w, h] に変換する"""
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    dets = []
    for r in results:
        inst = r['pred_instances']
        boxes, scores = _to_numpy(inst['bboxes']), _to_numpy(inst['scores'])
        labels = _to_numpy(inst['labels'])
        for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
            dets.append({'image_id': int(r['img_id']), 'category_id': label2cat[int(l)],
                         'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                         'score': float(s)})
    return dets


def prepare_dets_for_tide(det_path, gt_path, out_dir):
    """検出結果（.json か .pkl）を、tidecv が読める COCO 形式の json にしてパスを返す。
    .json はそのまま使い、.pkl は変換したものを out_dir に保存する"""
    ext = os.path.splitext(det_path)[1].lower()
    if ext == '.json':
        return det_path
    if ext != '.pkl':
        raise ValueError(f'検出結果は .json か .pkl にしてください: {det_path}')
    dets = pkl_to_coco_dets(det_path, label_to_category_id(gt_path))
    out_path = os.path.join(out_dir, 'dets_from_pkl.json')
    with open(out_path, 'w') as f:
        json.dump(dets, f)
    print(f'pkl を COCO 形式に変換しました: {out_path}（検出 {len(dets)} 個）')
    return out_path


def load_for_tide(gt_path, det_path, out_dir):
    """正解と検出を tidecv の形式で1回だけ読み込む（閾値を変えて何度も使い回す）"""
    gt_path = prepare_gt_for_tide(gt_path, out_dir)
    det_path = prepare_dets_for_tide(det_path, gt_path, out_dir)
    return datasets.COCO(path=gt_path), datasets.COCOResult(path=det_path)


def eval_one_threshold(gt, preds, pos_thr, bg_thr):
    """1つの閾値で TIDE にかけ、{列名: 値} を返す。
    閾値ごとに TIDE を作り直し、前の閾値の結果をメモリに残さない"""
    tide = TIDE()
    tide.evaluate(gt, preds, mode=TIDE.BOX, name='run',
                  pos_threshold=pos_thr,          # これ以上の IoU で正解とみなす
                  background_threshold=bg_thr)    # これ未満の IoU は背景への誤検出とみなす
    main, special = tide.get_main_errors()['run'], tide.get_special_errors()['run']
    row = {'AP': tide.runs['run'].ap}
    row.update({k: main.get(k, float('nan')) for k in MAIN_ERRORS})
    row.update({k: special.get(k, float('nan')) for k in SPECIAL_ERRORS})
    return row


def average_over_thresholds(per_thr):
    """閾値ごとの結果 {閾値: {列名: 値}} を、閾値間で平均した {列名: 値} を返す。
    TIDE 公式の evaluate_range は誤りの内訳を1つの閾値でしか出さないので、この平均は独自の拡張"""
    return {c: float(np.mean([row[c] for row in per_thr.values()])) for c in COLS}


if __name__ == '__main__':
    # 動作確認：pkl の形を模した作り物のデータで、xyxy → xywh の変換と番号の対応を確かめる
    import tempfile
    fake = [{'img_id': 7, 'pred_instances': {'bboxes': np.array([[10., 20., 40., 60.]]),
                                             'scores': np.array([0.9]), 'labels': np.array([1])}}]
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        pickle.dump(fake, f)
    print(pkl_to_coco_dets(f.name, {0: 1, 1: 2}))
    # 期待値: [{'image_id': 7, 'category_id': 2, 'bbox': [10.0, 20.0, 30.0, 40.0], 'score': 0.9}]