"""
【集計部分】検出結果（COCO 形式の *.bbox.json）と正解を対応付け、
検出ごとに「信頼度スコア・IoU・相手の物体のサイズ区分・クラス一致」を求める。
描画はしない（plot_score_iou.py から呼ぶ）。

割り当てのルール（以前と同じ）:
  - 各検出について、クラスを無視して IoU が最大の GT を「相手」とする
  - 同じ GT を相手にする検出が複数あれば、スコアが最大の1つだけ残し、残りは重複として省く
  - どの GT とも重ならない検出（IoU = 0）は相手なしとして残す
サイズ区分は相手の GT の面積（json の area）で COCO の境界（32², 96²）を使う。
変形後の座標の正解で IoU を測るときも、load_gt の size_gt_path で面積だけ元画像のものにできる。
件数表（検出枠数・クラス正解/不正解・IoU 閾値での内訳・IoU の平均と中央値を、
「全検出 / スコアで絞った版」×「all / small / medium / large」ごとに）の数え上げもここで行う
（summary_block, build_summaries。表示と CSV 保存はメイン側）。

置き場所: projects/Edge_poisson_asap/tools/score_iou_match.py
動作確認: python projects/Edge_poisson_asap/tools/score_iou_match.py （ダミーデータで割り当てを確かめる）
"""
import json
import os
from collections import defaultdict

import numpy as np

# COCO の物体サイズの境界（元画像での面積 [ピクセル数]）
SMALL_MAX_AREA = 32 ** 2
MEDIUM_MAX_AREA = 96 ** 2

# サイズ区分の番号（相手の GT がいない検出は SIZE_NONE）
SIZE_NONE, SIZE_SMALL, SIZE_MEDIUM, SIZE_LARGE = -1, 0, 1, 2
SIZE_NAMES = {SIZE_SMALL: "small", SIZE_MEDIUM: "medium",
              SIZE_LARGE: "large", SIZE_NONE: "no GT"}

# 図に使うデータの選び方（--subset の選択肢）
#   all   : 重複を省いた全検出（相手なし・クラス不一致も含む）
#   match : そのうち相手の GT とクラスが一致した検出だけ
SUBSET_ALL, SUBSET_MATCH = "all", "match"
SUBSETS = (SUBSET_ALL, SUBSET_MATCH)

# 件数表のサイズの区切り（None = サイズで絞らない。相手なしの検出は all にだけ入る）
SIZE_GROUPS = [(None, "all"), (SIZE_SMALL, "small"), (SIZE_MEDIUM, "medium"), (SIZE_LARGE, "large")]


def _group_by_image(items, keys):
    """json の要素のリストを画像 id ごとにまとめ、各キーを numpy 配列にする"""
    grouped = defaultdict(lambda: {k: [] for k in keys})
    for it in items:
        g = grouped[it["image_id"]]
        for k in keys:
            g[k].append(it[k])
    out = {}
    for img_id, g in grouped.items():
        out[img_id] = {
            "bbox": np.asarray(g["bbox"], dtype=np.float64).reshape(-1, 4),   # [x, y, w, h]
            **{k: np.asarray(g[k]) for k in keys if k != "bbox"},
        }
    return out


def load_gt(gt_path, size_gt_path=None):
    """
    正解 json を読み、画像 id ごとに GT の箱・クラス・面積をまとめる。
    gt_path     : IoU を測る正解（検出結果と同じ座標系のもの）
    size_gt_path: サイズ区分に使う面積を別の json から取るときに指定する（省略時は gt_path の面積）。
                  変形後の座標の正解で IoU を測るときに、サイズ区分は元画像の面積のままにするため。
                  2つの json はアノテーション id で対応させる（変形後の json は元の id を残している）
    """
    with open(gt_path) as f:
        anns = json.load(f)["annotations"]
    if size_gt_path is not None and os.path.abspath(size_gt_path) != os.path.abspath(gt_path):
        with open(size_gt_path) as f:
            area_by_id = {a["id"]: a["area"] for a in json.load(f)["annotations"]}
        missing = [a["id"] for a in anns if a["id"] not in area_by_id]
        if missing:
            raise KeyError(f"{size_gt_path} にないアノテーション id が {len(missing)} 個あります"
                           f"（例: {missing[:5]}）")
        # 面積だけ元画像のものに置き換える（箱は gt_path の座標のまま）
        anns = [{**a, "area": area_by_id[a["id"]]} for a in anns]
    return _group_by_image(anns, ["bbox", "category_id", "area"])


def load_dets(pred_path):
    """*.bbox.json（検出結果のリスト）を読み、画像 id ごとに箱・クラス・スコアをまとめる"""
    with open(pred_path) as f:
        dets = json.load(f)
    if len(dets) == 0:
        raise ValueError(f"{pred_path} に検出が1つもありません")
    return _group_by_image(dets, ["bbox", "category_id", "score"])


def iou_matrix(a, b):
    """
    [x, y, w, h] 形式の箱どうしの IoU を全組み合わせで求める（クラスは見ない）
    a: (N, 4), b: (M, 4)  ->  (N, M)
    """
    ax1, ay1 = a[:, 0:1], a[:, 1:2]
    ax2, ay2 = ax1 + a[:, 2:3], ay1 + a[:, 3:4]
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = bx1 + b[:, 2], by1 + b[:, 3]
    # 重なり部分の幅と高さ（重ならなければ 0）
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = a[:, 2:3] * a[:, 3:4] + b[:, 2] * b[:, 3] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def size_category(areas):
    """面積から COCO のサイズ区分の番号を返す"""
    return np.where(areas < SMALL_MAX_AREA, SIZE_SMALL,
                    np.where(areas < MEDIUM_MAX_AREA, SIZE_MEDIUM, SIZE_LARGE))


def match_image(det, gt):
    """
    1枚分の検出と GT を割り当てる。
    Returns: (残した検出の dict{score, iou, size, cls_match}, 省いた重複の数)
    """
    n = len(det["score"])
    scores = det["score"].astype(np.float64)
    # GT がない画像では、すべて相手なし
    if gt is None or len(gt["bbox"]) == 0:
        return {"score": scores, "iou": np.zeros(n), "size": np.full(n, SIZE_NONE),
                "cls_match": np.zeros(n, dtype=bool)}, 0

    # 各検出の相手 = クラスを無視して IoU が最大の GT
    ious = iou_matrix(det["bbox"], gt["bbox"])
    best_gt = ious.argmax(axis=1)
    best_iou = ious[np.arange(n), best_gt]
    has_gt = best_iou > 0

    # 重複の除去: スコアの高い順に並べ、同じ相手を持つ検出のうち最初（スコア最大）だけ残す
    order = np.argsort(-scores, kind="stable")
    matched = order[has_gt[order]]                       # 相手のある検出（スコア降順）
    _, first = np.unique(best_gt[matched], return_index=True)
    keep = ~has_gt                                       # 相手のない検出はすべて残す
    keep[matched[first]] = True
    n_dup = int(has_gt.sum() - len(first))

    # 残した検出について、相手のサイズ区分とクラス一致を求める
    size = np.where(has_gt, size_category(gt["area"][best_gt]), SIZE_NONE)
    cls_match = has_gt & (det["category_id"] == gt["category_id"][best_gt])
    return {"score": scores[keep], "iou": best_iou[keep],
            "size": size[keep], "cls_match": cls_match[keep]}, n_dup


def match_all(gts, dets):
    """
    全画像で割り当てを行い、検出ごとの値を1本の配列につなげる。
    Returns: 配列の dict{score, iou, size, cls_match} と、画面表示用の集計
    """
    parts, n_dup_total = [], 0
    for img_id, det in dets.items():
        part, n_dup = match_image(det, gts.get(img_id))
        parts.append(part)
        n_dup_total += n_dup
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    raw_scores = np.concatenate([d["score"] for d in dets.values()]).astype(np.float64)
    stats = {
        "n_images": len(dets),                                   # 検出のある画像の枚数
        "n_det": len(raw_scores),                                # 重複を省く前の検出数
        "n_dup": n_dup_total,                                    # 省いた重複の数
        "n_kept": len(merged["score"]),                          # 重複を省いた後の検出数
        "n_match": int(merged["cls_match"].sum()),               # そのうちクラス一致の数
        "max_per_img": max(len(d["score"]) for d in dets.values()),
        "min_score": float(raw_scores.min()),   # json 内の最小スコア（test_cfg.score_thr の確認用）
        "raw_scores": raw_scores,      # 省く前の全検出のスコア（件数表の「検出枠数（重複を含む）」に使う）
    }
    return merged, stats


def select_subset(m, subset):
    """
    図に使う検出を選ぶ（以前の match_only=True に当たるのが subset="match"）。
    m      : match_all が返す検出ごとの配列の dict
    subset : "all" ならそのまま、"match" ならクラス一致の検出だけ（相手なしも消える）
    """
    if subset == SUBSET_ALL:
        return m
    if subset == SUBSET_MATCH:
        return {k: v[m["cls_match"]] for k, v in m.items()}
    raise ValueError(f"subset は {SUBSETS} のどれかにしてください: {subset}")


def summary_block(m, raw_scores, iou_thr, score_thr=None, size=None):
    """
    件数表の1区切り分（スコアの絞り込み × サイズ）の行を作る（以前の score_iou_summary.py と同じ項目）。
    m          : match_all が返す検出ごとの配列（重複を省いた後、select_subset で絞る前）
    raw_scores : 重複を省く前の全検出のスコア（「検出枠数（重複を含む）」に使う）
    score_thr  : None なら全検出、数値ならスコアがそれ以上の検出だけ
    size       : None なら全サイズ、サイズ区分の番号ならその大きさの GT を相手に持つ検出だけ
    Returns: [(項目名, 値), ...]（値が None の行は「-」と表示する）
    """
    # 区切りに入る検出を選ぶ
    keep = np.ones(len(m["score"]), dtype=bool)
    if score_thr is not None:
        keep &= m["score"] >= score_thr
    if size is not None:
        keep &= m["size"] == size
    has_gt = (m["size"] != SIZE_NONE)[keep]      # 相手の GT がいる（IoU > 0）
    match = m["cls_match"][keep]                 # 相手の GT とクラスが一致
    iou = m["iou"][keep]
    high = iou >= iou_thr                        # IoU が閾値以上（位置が合っている）

    # 重複を含む検出数と相手なしの数は、サイズ（相手の GT）が決まらないので全サイズのときだけ出す
    all_size = size is None
    n_raw = int((raw_scores >= score_thr).sum()) if score_thr is not None else len(raw_scores)
    match_iou = iou[match]
    t = f"{iou_thr:g}"
    return [
        ("検出枠数（重複を含む）", n_raw if all_size else None),
        ("重複を省いた検出枠数", int(keep.sum())),
        ("クラス正解", int(match.sum())),
        ("クラス不正解（相手あり）", int((has_gt & ~match).sum())),
        (f"  うち IoU>={t}（位置は合うがクラス違い）", int((has_gt & ~match & high).sum())),
        ("相手なし（IoU=0）", int((~has_gt).sum()) if all_size else None),
        (f"IoU>={t} かつクラス正解", int((match & high).sum())),
        (f"IoU<{t} かつクラス正解（位置ずれ）", int((match & ~high).sum())),
        ("クラス正解の IoU 平均", float(match_iou.mean()) if len(match_iou) else None),
        ("クラス正解の IoU 中央値", float(np.median(match_iou)) if len(match_iou) else None),
    ]


def build_summaries(m, raw_scores, iou_thr, score_thrs):
    """
    件数表の全区切り（全検出 / score>=t × all / small / medium / large）を作る（1手法分）。
    重複の除去は全検出で行った後にスコアで絞る。残す検出は GT ごとにスコア最大のものなので、
    先にスコアで絞ってから重複を除いても結果は同じになる。
    Returns: [(スコアの区切り名, サイズ名, summary_block の結果), ...]
    """
    blocks = []
    for score_thr in [None] + list(score_thrs):
        s_name = "全検出" if score_thr is None else f"score>={score_thr:g}"
        for size, size_name in SIZE_GROUPS:
            blocks.append((s_name, size_name, summary_block(m, raw_scores, iou_thr, score_thr, size)))
    return blocks


if __name__ == "__main__":
    # 動作確認: GT 2個・検出 4個の画像1枚で割り当てのルールを確かめる
    gt = {"bbox": np.array([[0, 0, 10, 10], [100, 100, 200, 200]], dtype=np.float64),
          "category_id": np.array([1, 2]),
          "area": np.array([100.0, 40000.0])}          # small と large
    det = {"bbox": np.array([[0, 0, 10, 10],          # GT0 にぴったり（クラス一致）
                             [1, 1, 10, 10],          # GT0 の重複（スコアが低いので省かれる）
                             [100, 100, 100, 200],    # GT1 に半分重なる（クラス不一致）
                             [500, 500, 10, 10]],     # どれとも重ならない（相手なし）
                            dtype=np.float64),
           "category_id": np.array([1, 1, 3, 1]),
           "score": np.array([0.9, 0.5, 0.7, 0.3])}
    m, n_dup = match_image(det, gt)
    print("割り当て結果:", {k: v.tolist() for k, v in m.items()}, "重複:", n_dup)
    assert n_dup == 1
    assert np.allclose(m["iou"], [1.0, 0.5, 0.0])
    assert m["size"].tolist() == [SIZE_SMALL, SIZE_LARGE, SIZE_NONE]
    assert m["cls_match"].tolist() == [True, False, False]
    # クラス一致だけを選ぶと1個になる
    assert len(select_subset(m, SUBSET_MATCH)["score"]) == 1
    # 件数表: 全検出・全サイズでは 重複を含む4 / 省いた後3 / 正解1 / 不正解1（うち IoU>=0.5 が1）/ 相手なし1
    rows = dict(summary_block(m, det["score"], 0.5))
    print("件数表（全検出・all）:", rows)
    assert rows["検出枠数（重複を含む）"] == 4 and rows["重複を省いた検出枠数"] == 3
    assert rows["クラス正解"] == 1 and rows["クラス不正解（相手あり）"] == 1
    assert rows["  うち IoU>=0.5（位置は合うがクラス違い）"] == 1 and rows["相手なし（IoU=0）"] == 1
    # small だけにすると正解の1件だけになり、相手なしは数えない（-）
    rows_s = dict(summary_block(m, det["score"], 0.5, size=SIZE_SMALL))
    assert rows_s["重複を省いた検出枠数"] == 1 and rows_s["相手なし（IoU=0）"] is None
    # score>=0.5 で絞ると、重複を含む数は 0.9, 0.5, 0.7 の3個、省いた後は 0.9, 0.7 の2個
    rows_h = dict(summary_block(m, det["score"], 0.5, score_thr=0.5))
    assert rows_h["検出枠数（重複を含む）"] == 3 and rows_h["重複を省いた検出枠数"] == 2
    # 区切りの数: (全検出 + 閾値1つ) × 4サイズ = 8
    assert len(build_summaries(m, det["score"], 0.5, [0.3])) == 8
    print("OK")