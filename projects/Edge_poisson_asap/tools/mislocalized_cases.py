"""位置ずれの検出（クラスは正しいが IoU が低い検出）を集めるモジュール。

割り当ての規則は plot_score_iou.py と同じにしている：
  - クラスを無視して、IoU が最大の正解を相手にする
  - 同じ正解を指す検出が複数あれば、スコア最大の1つだけ残す（残りは重複として除く）
  - IoU=0 の検出は相手なし

割り当てをどちらの座標で行うか（match_space）:
  - "orig"    : 元の座標で割り当てる（従来どおり。plot_score_iou.py の評価と同じ見方）。
                変形後の座標の IoU は、元の座標で選んだ「同じ正解」との IoU を測る
  - "deformed": 変形後の座標で割り当てる。元の座標の IoU は、変形後の座標で選んだ「同じ正解」
                （annotation id が同じもの）との IoU を測る。
                変形後の座標では合っていたのに、元の座標へ戻すときに「裏返り」などで大きく崩れた検出を
                見るときに使う。元の座標で割り当てると、崩れた検出は IoU=0 で「相手なし」になったり、
                隣の別の正解に割り当てられたりして、目的の検出が集まらないため

集計（summary_lines）:
  クラス正解で割り当てられた検出（分母）のうち、元の座標の IoU が対象範囲のものを、
  変形後の座標の IoU の区間（例: >=0.75 / 0.50-0.75 / <0.50 / 不明）ごとに、サイズ別・クラス別に数える

json の読み込み・サイズ区分・IoU 行列・画像ごとのまとめ・表示幅は eval_common.py（ほかのツールと共通）。
以前この中にあった名前（load_json, size_label, iou_matrix, index_by_image, text_width）は、
ほかのコードから読み込めるよう、ここでも同じ名前で使えるようにしてある。
"""
import os
from collections import defaultdict

import numpy as np

from eval_common import group_by_image, iou_matrix, load_json, size_label, text_width  # noqa: F401

# 以前の名前（画像ごとにまとめる関数）。中身は eval_common.group_by_image と同じ
index_by_image = group_by_image

# 割り当てに使える座標の種類
MATCH_SPACES = ("orig", "deformed")
# サイズ区分の表示順（"all" は全体）
SIZES = ("all", "small", "medium", "large")


def pair_iou(det, gt):
    """検出1つと正解1つの IoU。どちらかがなければ NaN（不明）を返す"""
    if det is None or gt is None:
        return float('nan')
    return float(iou_matrix([det['bbox']], [gt['bbox']])[0, 0])


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


def match_in_orig(dets_o, pairs, gts_o_img, gt_def_by_id):
    """元の座標で割り当てる（従来の方法）。
    戻り値: (元座標の検出, 変形後の検出, 元の正解, 変形後の正解, 元の座標の IoU) のリスト"""
    out = []
    for i, (j, iou_o) in match_one_image(dets_o, gts_o_img).items():
        gt_o = gts_o_img[j]
        # 変形後の座標では割り当て直さず、annotation id が同じ「同じ正解」を使う
        out.append((dets_o[i], pairs.get(i), gt_o, gt_def_by_id.get(gt_o['id']), iou_o))
    return out


def match_in_deformed(dets_o, dets_d, pairs, gts_d_img, gt_orig_by_id):
    """変形後の座標で割り当てる。元の座標の IoU は、同じ正解（annotation id が同じ）との IoU を測る。
    戻り値: (元座標の検出, 変形後の検出, 元の正解, 変形後の正解, 元の座標の IoU) のリスト
            （元の座標の IoU は、元座標の検出か元の正解が見つからなければ NaN）"""
    # pair_detections の結果（元 → 変形後）を逆向き（変形後 → 元）にする。
    # 値は検出の dict そのものなので、オブジェクトの id で引けるようにする
    rev = {id(d): dets_o[i] for i, d in pairs.items() if d is not None}
    out = []
    for k, (j, _iou_d) in match_one_image(dets_d, gts_d_img).items():
        det_d, gt_d = dets_d[k], gts_d_img[j]
        det_o, gt_o = rev.get(id(det_d)), gt_orig_by_id.get(gt_d['id'])
        out.append((det_o, det_d, gt_o, gt_d, pair_iou(det_o, gt_o)))
    return out


def collect_cases(gt_orig, gt_def, det_orig, det_def, score_thr=0.01, iou_min=0.0, iou_max=0.5,
                  match_space="orig"):
    """クラス正解かつ iou_min <= IoU < iou_max（元座標）の検出を集め、
    同じ正解に対する非一様縮小後の座標での IoU も添える。
    match_space: 割り当てをどちらの座標で行うか（"orig" / "deformed"。モジュールの説明を参照）
    戻り値: (検出ごとの dict のリスト, 並び順で対応付けられなかった画像の数)"""
    if match_space not in MATCH_SPACES:
        raise ValueError(f"match_space は {MATCH_SPACES} のどれか: {match_space}")
    img_map = map_image_ids(gt_orig, gt_def)
    gts_o = index_by_image(gt_orig['annotations'])
    gts_d = index_by_image(gt_def['annotations'])
    # 元と変形後の正解は annotation id で対応付ける（make_deformed_dataset.py は id を変えない）
    gt_orig_by_id = {a['id']: a for a in gt_orig['annotations']}
    gt_def_by_id = {a['id']: a for a in gt_def['annotations']}
    # 表1・2と同じく、スコアで絞ってから割り当てる（重複の除き方が閾値で変わるため）
    dets_o_all = index_by_image(d for d in det_orig if d['score'] >= score_thr)
    dets_d_all = index_by_image(d for d in det_def if d['score'] >= score_thr)

    cases, n_fallback = [], 0
    for img_id, dets_o in dets_o_all.items():
        img_id_def = img_map.get(img_id)
        dets_d = dets_d_all.get(img_id_def, [])
        pairs, ordered = pair_detections(dets_o, dets_d)
        n_fallback += int(not ordered)
        if match_space == "orig":
            matched = match_in_orig(dets_o, pairs, gts_o.get(img_id, []), gt_def_by_id)
        else:
            matched = match_in_deformed(dets_o, dets_d, pairs, gts_d.get(img_id_def, []), gt_orig_by_id)

        for det_o, det_d, gt_o, gt_d, iou_o in matched:
            # クラスの判定は、割り当てた側の座標の検出と正解で行う（どちらの座標でもクラスは同じ）
            det, gt = (det_o, gt_o) if match_space == "orig" else (det_d, gt_d)
            if det_o is None or gt_o is None or det['category_id'] != gt['category_id']:
                continue                               # 対応が取れない、またはクラス違い
            if not (iou_min <= iou_o < iou_max):
                continue                               # 元の座標の IoU が対象の範囲外
            area = gt_o.get('area', gt_o['bbox'][2] * gt_o['bbox'][3])
            cases.append(dict(
                image_id=img_id, image_id_def=img_id_def, ann_id=gt_o['id'],
                category_id=gt_o['category_id'], size=size_label(area), score=det_o['score'],
                iou_orig=iou_o, iou_def=pair_iou(det_d, gt_d),
                gt_bbox_orig=gt_o['bbox'], det_bbox_orig=det_o['bbox'],
                gt_bbox_def=gt_d['bbox'] if gt_d is not None else None,
                det_bbox_def=det_d['bbox'] if det_d is not None else None))
    return cases, n_fallback


# ===========================================================================
# 集計（件数の表）
# ===========================================================================
def filter_orig_range(cases, iou_min, iou_max):
    """元の座標の IoU が iou_min 以上 iou_max 未満のものだけを返す（NaN は含めない）"""
    return [c for c in cases if iou_min <= c['iou_orig'] < iou_max]


def def_bin_labels(edges):
    """変形後の IoU の区間の名前を、値の大きい順に返す（最後は「不明」）。
    edges: 区間の境目（小さい順）。例: [0.5, 0.75] -> ['>=0.75', '0.50-0.75', '<0.50', '不明']"""
    e = list(edges)
    labels = [f"<{e[0]:.2f}"] + [f"{a:.2f}-{b:.2f}" for a, b in zip(e, e[1:])] + [f">={e[-1]:.2f}"]
    return labels[::-1] + ["不明"]


def def_bin_index(v, edges):
    """変形後の IoU v が入る区間の番号（def_bin_labels と同じ、値の大きい順）。NaN は最後（不明）。
    境目の値は上の区間に入れる（0.75 は >=0.75、0.50 は 0.50-0.75）"""
    n = len(edges) + 1                             # NaN 以外の区間の数
    if np.isnan(v):
        return n
    return n - 1 - int(np.searchsorted(edges, v, side='right'))


def format_table(header, rows):
    """表を、列ごとに幅をそろえた文字列の行のリストにする（1列目は左寄せ、ほかは右寄せ）"""
    table = [header] + rows
    widths = [max(text_width(r[k]) for r in table) for k in range(len(header))]   # 列ごとの幅

    def pad(v, w, left):
        # 全角文字を含んでも列がそろうよう、表示幅で空白を足す
        space = ' ' * (w - text_width(v))
        return str(v) + space if left else space + str(v)
    return ["  ".join(pad(v, w, k == 0) for k, (v, w) in enumerate(zip(r, widths))) for r in table]


def _pct(n, total):
    """割合を「12.3%」の形の文字列にする（分母が 0 なら「-」）"""
    return f"{100 * n / total:.1f}%" if total else "-"


def count_rows(matched, cases, edges, groups, group_of):
    """グループごとに、割り当て数・対象数・変形後の IoU の区間ごとの数・平均スコアを数えた表の行を返す。
    matched : クラス正解で割り当てられた検出すべて（割合の分母）
    cases   : そのうち元の座標の IoU が対象範囲のもの
    groups  : 表に出すグループ名の並び（"all" は全体）
    group_of: case -> グループ名 を返す関数"""
    rows = []
    for g in groups:
        m = [c for c in matched if g == "all" or group_of(c) == g]   # このグループの割り当て数（分母）
        cs = [c for c in cases if g == "all" or group_of(c) == g]    # このグループの対象
        idx = np.array([def_bin_index(c['iou_def'], edges) for c in cs], dtype=int)
        cnt = np.bincount(idx, minlength=len(edges) + 2)            # 区間ごとの数（最後は不明）
        row = [g, len(m), f"{len(cs)} ({_pct(len(cs), len(m))})"]
        row += [f"{n} ({_pct(n, len(cs))})" for n in cnt]          # 区間の割合は対象数に対するもの
        row.append(f"{np.mean([c['score'] for c in cs]):.3f}" if cs else "-")
        rows.append(row)
    return rows


def summary_lines(matched, cases, edges, iou_min, iou_max, cat_names):
    """件数の集計を、表示・保存用の文字列の行のリストにして返す。
    matched  : クラス正解で割り当てられた検出すべて / cases: そのうち元の座標の IoU が対象範囲のもの
    edges    : 変形後の IoU の区間の境目（小さい順）
    cat_names: カテゴリ id -> クラス名"""
    labels = def_bin_labels(edges)
    header = ["", "割り当て数", f"元IoU<{iou_max:.2f}"] + [f"変形後{l}" for l in labels] + ["平均スコア"]
    lines = [f"■ 元の座標の IoU が {iou_min:.2f} 以上 {iou_max:.2f} 未満の検出の件数と、変形後の座標での IoU の内訳",
             "  割り当て数: クラス正解で正解に割り当てられた検出の数（割合の分母）",
             f"  元IoU<{iou_max:.2f}: 対象の件数（括弧は割り当て数に対する割合）",
             "  変形後〜: 対象のうち、変形後の座標の IoU がその区間のもの（括弧は対象に対する割合）",
             "  不明: 変形後の検出か正解の対応が取れなかったもの / 平均スコア: 対象の検出スコアの平均",
             "", "[サイズ別]（サイズは元の座標の正解の面積で決める）"]
    lines += format_table(header, count_rows(matched, cases, edges, SIZES, lambda c: c['size']))

    # クラス別（割り当てが1件以上あるクラスだけ。表のグループ名はクラス名）
    present = sorted({c['category_id'] for c in matched})
    names = {cid: cat_names.get(cid, str(cid)) for cid in present}
    lines += ["", "[クラス別]"]
    lines += format_table(header, count_rows(matched, cases, edges, ["all"] + [names[i] for i in present],
                                             lambda c: names[c['category_id']]))

    # いちばん上の区間（変形後では合っていた）のうち、元の座標では完全に外れたもの
    top = [c for c in cases if def_bin_index(c['iou_def'], edges) == 0]
    zero = sum(c['iou_orig'] == 0 for c in top)
    drop = [c['iou_def'] - c['iou_orig'] for c in top]
    lines += ["", f"変形後{labels[0]} の {len(top)} 件のうち、元の座標で IoU=0（完全に外れた）: {zero} 件"
                  + (f" / IoU の落ち幅（変形後−元）の平均 {np.mean(drop):.3f}、最大 {np.max(drop):.3f}" if top else "")]
    return lines


if __name__ == '__main__':
    # 動作確認1：小さな作り物のデータで割り当てと IoU を確かめる
    gts = [dict(id=1, bbox=[0, 0, 10, 10], category_id=1, area=100),
           dict(id=2, bbox=[50, 50, 10, 10], category_id=2, area=100)]
    dets = [dict(bbox=[2, 0, 10, 10], score=0.9, category_id=1),    # 正解1と IoU 0.667
            dict(bbox=[5, 0, 10, 10], score=0.5, category_id=1),    # 正解1の重複（スコアが低いので除かれる）
            dict(bbox=[55, 55, 10, 10], score=0.8, category_id=2),  # 正解2と IoU 0.143
            dict(bbox=[200, 200, 5, 5], score=0.7, category_id=1)]  # 相手なし
    print(match_one_image(dets, gts))  # 期待値: {0: (0, 0.667), 2: (1, 0.143)}

    # 動作確認2：割り当ての座標による違い。
    # 変形後の座標では正解とぴったり合っている検出が、元の座標へ戻すと正解から完全に外れた
    # （裏返りなどで崩れた）場合を作る
    img = dict(id=7, file_name='a.jpg')
    g_o = dict(id=11, image_id=7, bbox=[100, 100, 10, 10], category_id=1, area=100)   # 元の正解
    g_d = dict(id=11, image_id=7, bbox=[50, 50, 5, 5], category_id=1, area=25)        # 変形後の正解
    d_d = dict(image_id=7, bbox=[50, 50, 5, 5], score=0.9, category_id=1)            # 変形後: IoU 1.0
    d_o = dict(image_id=7, bbox=[130, 100, 10, 10], score=0.9, category_id=1)        # 元: IoU 0（外れた）
    gt_o_json = dict(images=[img], annotations=[g_o])
    gt_d_json = dict(images=[img], annotations=[g_d])
    for space in MATCH_SPACES:
        cs, nf = collect_cases(gt_o_json, gt_d_json, [d_o], [d_d], 0.3, 0.0, 0.5, match_space=space)
        print(f"match_space={space:8s}: {len(cs)} 件",
              [(c['ann_id'], round(c['iou_orig'], 3), round(c['iou_def'], 3)) for c in cs])
    # 期待値: orig は 0 件（元の座標で IoU=0 なので「相手なし」）
    #         deformed は 1 件 [(11, 0.0, 1.0)]

    # 動作確認3：区間の番号（境目の値は上の区間に入る）と、集計の表
    edges = [0.5, 0.75]
    print(def_bin_labels(edges), [def_bin_index(v, edges) for v in (0.9, 0.75, 0.6, 0.5, 0.2, float('nan'))])
    # 期待値: ['>=0.75', '0.50-0.75', '<0.50', '不明'] [0, 0, 1, 1, 2, 3]
    fake = [dict(category_id=1, size='small', score=0.9, iou_orig=0.0, iou_def=0.9),
            dict(category_id=1, size='small', score=0.5, iou_orig=0.3, iou_def=0.6),
            dict(category_id=2, size='medium', score=0.7, iou_orig=0.4, iou_def=0.2),
            dict(category_id=2, size='medium', score=0.8, iou_orig=0.8, iou_def=0.85)]   # 対象外（元 IoU>=0.5）
    for line in summary_lines(fake, filter_orig_range(fake, 0.0, 0.5), edges, 0.0, 0.5,
                              {1: 'car', 2: 'van'}):
        print(line)
    # 期待値: all は 割り当て 4、元IoU<0.50 が 3 (75.0%)、>=0.75 / 0.50-0.75 / <0.50 が 1 件ずつ