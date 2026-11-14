"""magnification_lib.py
変形前（元座標）と変形後の bbox（COCO 形式）から、物体ごとの拡大率を計算する関数群。

拡大率の定義（物体 i ごと）:
    面積倍率 m_area = (w' * h') / (w * h * sx * sy)
    長さ倍率 m_len  = sqrt(m_area)          … 辺の倍率（縦横の幾何平均）
    w', h' : 変形後の bbox の幅・高さ
    w, h   : 元座標の bbox の幅・高さ
    sx, sy : 基準となる一様縮小の倍率（baseline='none' なら 1）
"""
import json
import math

import numpy as np

# COCO 基準のサイズ区分の境界（元座標の面積、単位は画素^2）
COCO_SMALL_MAX = 32 ** 2   # これ未満を small
COCO_MEDIUM_MAX = 96 ** 2  # これ未満を medium、以上を large
SIZE_NAMES = ('small', 'medium', 'large')

# 統計を取る対象の指標名
METRICS = ('area_ratio', 'length_ratio')


def load_coco_like(path):
    """COCO 形式の json を読み、(annotation のリスト, image_id→(幅, 高さ) の辞書) を返す。

    {"images": [...], "annotations": [...]} の形式と、
    annotation だけを並べたリスト形式の両方に対応する（後者は画像サイズなし）。
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        anns = data.get('annotations', [])  # 物体ごとの情報
        imgs = {im['id']: (im['width'], im['height'])  # 画像ごとのサイズ
                for im in data.get('images', [])}
    else:
        anns = data
        imgs = {}
    return anns, imgs


def coco_size_category(area):
    """元座標の面積から COCO 基準のサイズ区分名（small/medium/large）を返す。"""
    if area < COCO_SMALL_MAX:
        return 'small'
    if area < COCO_MEDIUM_MAX:
        return 'medium'
    return 'large'


def uniform_scale(orig_size, warp_size, baseline, frame):
    """基準にする一様縮小の倍率 (sx, sy) を返す。

    baseline='frame': 元画像を枠 frame=(幅, 高さ) に縦横比を保って収めたときの倍率（sx=sy）
    baseline='image': 変形後画像サイズ / 元画像サイズ（軸ごと。json の images の値を使う）
    baseline='none' : 1（元座標との生の比）
    """
    if baseline == 'none':
        return 1.0, 1.0
    if orig_size is None:
        raise ValueError('元の json に images（width/height）がないため基準倍率を計算できません')
    ow, oh = orig_size  # 元画像の幅・高さ
    if baseline == 'frame':
        s = min(frame[0] / ow, frame[1] / oh)  # 枠に収まる最大の倍率
        return s, s
    if baseline == 'image':
        if warp_size is None:
            raise ValueError('変形後の json に images（width/height）がありません')
        return warp_size[0] / ow, warp_size[1] / oh
    raise ValueError(f'未知の baseline: {baseline}')


def make_record(a, b, sx, sy, size_source):
    """元の annotation a と変形後の annotation b から、1 物体分の結果（辞書）を作る。"""
    w, h = a['bbox'][2], a['bbox'][3]      # 元座標の幅・高さ
    w2, h2 = b['bbox'][2], b['bbox'][3]    # 変形後の幅・高さ
    # サイズ区分に使う面積（json の area フィールドを優先、なければ bbox から）
    if size_source == 'area' and 'area' in a:
        size_area = a['area']
    else:
        size_area = w * h
    area_ratio = (w2 * h2) / (w * h * sx * sy)
    return {
        'ann_id': a['id'],
        'image_id': a['image_id'],
        'category_id': a.get('category_id'),
        'size': coco_size_category(size_area),
        'size_area': size_area,
        'w': w, 'h': h, 'w_warp': w2, 'h_warp': h2,
        'sx': sx, 'sy': sy,
        'area_ratio': area_ratio,
        'length_ratio': math.sqrt(area_ratio),
        'w_ratio': w2 / (w * sx),   # 横方向だけの倍率（歪みの確認用）
        'h_ratio': h2 / (h * sy),   # 縦方向だけの倍率（歪みの確認用）
    }


def compute_records(orig_path, warp_path, baseline='frame', frame=(800, 800),
                    size_source='area', skip_crowd=True, min_wh=1e-6):
    """2 つの json の annotation を id で対応付け、物体ごとの拡大率のリストを返す。

    戻り値: (records, skipped)
        records : make_record の辞書のリスト
        skipped : 除外した物体の数（理由ごと）
    """
    orig_anns, orig_imgs = load_coco_like(orig_path)
    warp_anns, warp_imgs = load_coco_like(warp_path)
    warp_by_id = {b['id']: b for b in warp_anns}  # 変形後を annotation id で引く辞書
    skipped = {'unmatched': 0, 'image_mismatch': 0, 'crowd': 0, 'degenerate': 0}
    records = []
    for a in orig_anns:
        b = warp_by_id.get(a['id'])
        if b is None:                       # 変形後の json に同じ id がない
            skipped['unmatched'] += 1
            continue
        if b['image_id'] != a['image_id']:  # id は一致したが画像が違う（対応付けの誤り）
            skipped['image_mismatch'] += 1
            continue
        if skip_crowd and a.get('iscrowd', 0):
            skipped['crowd'] += 1
            continue
        # 幅・高さがほぼ 0 の枠は比が発散するので除外
        if min(a['bbox'][2], a['bbox'][3], b['bbox'][2], b['bbox'][3]) <= min_wh:
            skipped['degenerate'] += 1
            continue
        sx, sy = uniform_scale(orig_imgs.get(a['image_id']),
                               warp_imgs.get(b['image_id']), baseline, frame)
        records.append(make_record(a, b, sx, sy, size_source))
    return records, skipped


def summarize(values):
    """値の配列から代表値（件数・平均・幾何平均・中央値・標準偏差・最小・最大）を返す。"""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {'count': 0}
    return {
        'count': int(v.size),
        'mean': float(v.mean()),                    # ZR に相当（算術平均）
        'geo_mean': float(np.exp(np.log(v).mean())),  # 比の平均として偏りにくい
        'median': float(np.median(v)),
        'std': float(v.std()),
        'min': float(v.min()),
        'max': float(v.max()),
    }


def summarize_records(records, metric):
    """全体とサイズ区分ごとの統計、および最大拡大率の物体を返す。"""
    groups = {'all': records}
    for name in SIZE_NAMES:
        groups[name] = [r for r in records if r['size'] == name]
    stats = {k: summarize([r[metric] for r in g]) for k, g in groups.items()}
    # 最大の拡大率の物体（全体とサイズ区分ごと）
    argmax = {k: max(g, key=lambda r: r[metric]) for k, g in groups.items() if g}
    return stats, argmax


if __name__ == '__main__':
    # 動作確認: 元画像 2000x1500 を 800x800 枠へ → 一様縮小の倍率は 0.4
    import os
    import tempfile
    orig = {'images': [{'id': 1, 'width': 2000, 'height': 1500}],
            'annotations': [
                {'id': 1, 'image_id': 1, 'bbox': [0, 0, 10, 10], 'area': 100},     # small
                {'id': 2, 'image_id': 1, 'bbox': [0, 0, 50, 50], 'area': 2500},    # medium
                {'id': 3, 'image_id': 1, 'bbox': [0, 0, 200, 100], 'area': 20000}]}  # large
    warp = {'images': [{'id': 1, 'width': 800, 'height': 600}],
            'annotations': [
                {'id': 1, 'image_id': 1, 'bbox': [0, 0, 8, 8]},    # 一様なら 4x4 → 面積4倍
                {'id': 2, 'image_id': 1, 'bbox': [0, 0, 20, 20]},  # 一様と同じ → 1倍
                {'id': 3, 'image_id': 1, 'bbox': [0, 0, 40, 40]}]}  # 一様なら 80x40 → 0.5倍
    tmp = tempfile.mkdtemp()
    p1, p2 = os.path.join(tmp, 'o.json'), os.path.join(tmp, 'w.json')
    for p, d in ((p1, orig), (p2, warp)):
        with open(p, 'w') as f:
            json.dump(d, f)
    recs, skp = compute_records(p1, p2)
    for r in recs:
        print(r['ann_id'], r['size'], round(r['area_ratio'], 4), round(r['length_ratio'], 4))
    assert [round(r['area_ratio'], 6) for r in recs] == [4.0, 1.0, 0.5]
    print('selftest OK', skp)