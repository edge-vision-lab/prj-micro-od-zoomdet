"""
座標の変換（正解ボックスを変形後へ、検出結果を元画像へ）

  - forward_points : 元画像の座標 -> 変形後（内容サイズ）の座標。順写像（ノードの dst）を使う
  - inverse_points : 変形後の座標 -> 元画像の座標。逆写像（map_x, map_y）を使う
  - forward_bbox / inverse_bbox : COCO 形式のボックス [x, y, w, h] を変換する。
        mode でボックスの作り方を選ぶ（下の「ボックスの作り方」を参照）
  - roundtrip_iou  : 元の正解 -> 変形後 -> 元画像 と往復させたボックスと、元の正解との IoU（方式の比較用）
  - draw_gt / save_gt_check : 確認用。元画像と非一様縮小後の画像に正解ボックスを描いて保存する

ボックスの作り方（mode）
  - "corner2": 左上・右下の2点だけを変換する（従来の方式。）
  - "mean"   : 4本の辺それぞれに n_per_side 個の点を等間隔に置いて変換し、
               左辺は x の平均、右辺は x の平均、上辺は y の平均、下辺は y の平均を、その辺の位置とする。
                 n_per_side = 2 -> 四隅の4点（左辺の x = 左上と左下の x の平均）
                 n_per_side = 3 -> 四隅 + 辺の中点の8点
                 n_per_side を大きくすると、曲がった辺の「平均の位置」に近づく
  - "minmax" : 同じ点の外接矩形（最小・最大）。必ず corner2 以上の大きさになるので、比較用の上限として使う

動作確認（projects/Edge_poisson_asap/ で実行。make_deformed_dataset.py で作ったデータを使う）:
    python -m edge_poisson_asap.box_transform \
        --orig-ann ../../data_ann/VisDrone/my_val.json \
        --orig-img ../../data/VisDrone/VisDrone2019-DET-val/images \
        --warp-dir ../../data_ann/VisDrone_poisson_asap/val --limit 3
  --orig-img にはフォルダ（名前順に --limit 枚）か、画像ファイル（複数可）を指定する
    例: --orig-img ../../data/VisDrone/VisDrone2019-DET-val/images/0000001_02999_d_0000005.jpg
  画像ごとに次の3枚を、このファイルと同じ場所の gt_check/ に保存する（--out-dir で変更可）
    {名前}_orig_gt.png    元画像に元の正解を描いたもの
    {名前}_warped_gt.png  非一様縮小後の画像に、変換後の正解を描いたもの
    {名前}_gt_compare.png 上の2枚を横に並べたもの（元画像は高さをそろえるため縮小して表示）
  ボックスの色: --color-mode single（既定。全部同じ色）/ class（カテゴリごとに色を分ける）
  ※ この確認は annotations.json に保存済みのボックスを描くだけなので、mode を変えて比べるときは
     make_deformed_dataset.py 側で mode を変えてデータを作り直す。

座標の約束（ボックスと画素の関係）
  - COCO のボックス [x, y, w, h] は「画素の端」を 0 とする座標（左上の画素の左上の角が (0, 0)）。
  - メッシュや map は「画素の中心」を整数とする座標（cv2 と同じ）。
  - 両者は 0.5 ずれている: 中心の座標 = 端の座標 - 0.5
  この関数群の入出力はすべて「画素の端」基準（COCO と同じ）で、中で中心基準に直して計算する。
"""
import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np

# 使えるボックスの作り方
BOX_MODES = ("corner2", "mean", "minmax")

# box_transform.py の先頭付近。正解の作成と逆変換で必ず同じ方式を使うため、1か所で決める
DEFAULT_MODE = "mean"      # ボックスの作り方
DEFAULT_N_PER_SIDE = 3     # 1本の辺に置く点の数

def _bilinear(grid, u, v):
    """
    (N, M, 2) の値の表 grid を、小数の位置 (列 u, 行 v) で双線形補間する（u, v は配列）。
    表の外（u < 0 など）は、いちばん近い端の値に「はみ出した分」をそのまま足して延長する。
    外周のノードは固定（変形前と同じ間隔）なので、端の外側は等倍とみなしてよい。
    """
    N, M, _ = grid.shape
    uc = np.clip(u, 0, M - 1)
    vc = np.clip(v, 0, N - 1)
    c0 = np.minimum(np.floor(uc).astype(int), M - 2)
    r0 = np.minimum(np.floor(vc).astype(int), N - 2)
    fu = (uc - c0)[:, None]
    fv = (vc - r0)[:, None]
    val = ((1 - fu) * (1 - fv) * grid[r0, c0] + fu * (1 - fv) * grid[r0, c0 + 1]
           + (1 - fu) * fv * grid[r0 + 1, c0] + fu * fv * grid[r0 + 1, c0 + 1])
    return val


def forward_points(pts, fmap):
    """
    元画像上の点 -> 変形後（内容サイズ）の点。
    pts : (K, 2) の (x, y)、画素の端基準（COCO と同じ）
    fmap: ForwardMap（in_size, content_size, dst を使う）

    考え方: ノードは元画像上で等間隔（一様縮小と同じ位置）に並んでいるので、
            点が「何番目のノードとノードの間か」は割り算で求まる。
            その位置で、各ノードの変形後の位置 dst を双線形補間する。
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    in_h, in_w = fmap.in_size
    out_h, out_w = fmap.content_size
    # 元画像の端基準の座標 -> ノード番号（小数）。ノード c の中心は端基準で (c + 0.5) * in_w / out_w
    u = pts[:, 0] * out_w / in_w - 0.5
    v = pts[:, 1] * out_h / in_h - 0.5
    q = _bilinear(fmap.dst, u, v)                 # 変形後の位置（中心基準）
    # 表の外にはみ出した分は等倍で延長（_bilinear の説明を参照）
    q[:, 0] += u - np.clip(u, 0, out_w - 1)
    q[:, 1] += v - np.clip(v, 0, out_h - 1)
    return q + 0.5                                # 中心基準 -> 端基準


def inverse_points(pts, map_x, map_y):
    """
    変形後（内容サイズ）の点 -> 元画像上の点。
    pts         : (K, 2) の (x, y)、画素の端基準
    map_x, map_y: build_inverse_map の出力（出力画素の中心ごとの、元画像の中心基準の座標）
    出力画素は等間隔に並んでいるので、map を双線形補間すればよい。
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    out_h, out_w = map_x.shape
    grid = np.stack([map_x, map_y], axis=-1).astype(float)
    u = pts[:, 0] - 0.5                           # 端基準 -> 中心基準
    v = pts[:, 1] - 0.5
    q = _bilinear(grid, u, v)
    # 表の外にはみ出した分は、端の画素の間隔（元画像の何画素分か）で延長する
    sx = (map_x[:, -1] - map_x[:, 0]).mean() / (out_w - 1)   # 外周固定なので端の間隔は一様縮小と同じ
    sy = (map_y[-1, :] - map_y[0, :]).mean() / (out_h - 1)
    q[:, 0] += (u - np.clip(u, 0, out_w - 1)) * sx
    q[:, 1] += (v - np.clip(v, 0, out_h - 1)) * sy
    return q + 0.5


# ===========================================================================
# ボックスの変換
# ===========================================================================
def _corners_to_bbox(p):
    """左上・右下の2点 (2K, 2) -> [x, y, w, h] (K, 4)。点の順序が入れ替わっても正しく作る"""
    tl, br = p[0::2], p[1::2]
    x1 = np.minimum(tl[:, 0], br[:, 0])
    y1 = np.minimum(tl[:, 1], br[:, 1])
    x2 = np.maximum(tl[:, 0], br[:, 0])
    y2 = np.maximum(tl[:, 1], br[:, 1])
    return np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)


def _bbox_to_corners(bboxes):
    """[x, y, w, h] (K, 4) -> 左上・右下を交互に並べた (2K, 2)"""
    b = np.asarray(bboxes, dtype=float).reshape(-1, 4)
    tl = b[:, :2]
    br = b[:, :2] + b[:, 2:]
    return np.stack([tl, br], axis=1).reshape(-1, 2)


def _side_points(bboxes, n_per_side):
    """
    [x, y, w, h] (K, 4) -> 4本の辺に等間隔に置いた点 (K, 4, n, 2)
      辺の並び: 0 = 左辺, 1 = 右辺, 2 = 上辺, 3 = 下辺
      各辺の両端は角の点（n_per_side = 2 なら角だけ、3 なら角 + 中点）
    """
    b = np.asarray(bboxes, dtype=float).reshape(-1, 4)
    x1, y1 = b[:, 0:1], b[:, 1:2]                 # (K, 1) 左上
    x2, y2 = x1 + b[:, 2:3], y1 + b[:, 3:4]       # (K, 1) 右下
    t = np.linspace(0.0, 1.0, n_per_side)[None]   # (1, n) 辺上の位置の割合
    xs = x1 + (x2 - x1) * t                       # (K, n) 上辺・下辺の x
    ys = y1 + (y2 - y1) * t                       # (K, n) 左辺・右辺の y
    one = np.ones_like(t)                         # (1, n) 定数の座標を n 個に広げる用
    left = np.stack([x1 * one, ys], axis=-1)      # (K, n, 2)
    right = np.stack([x2 * one, ys], axis=-1)
    top = np.stack([xs, y1 * one], axis=-1)
    bottom = np.stack([xs, y2 * one], axis=-1)
    return np.stack([left, right, top, bottom], axis=1)


def _aggregate_sides(q, mode):
    """
    変換後の辺上の点 q (K, 4, n, 2) -> [x, y, w, h] (K, 4)
    mode: "mean"   各辺の位置 = その辺の点の座標の平均（左右の辺は x、上下の辺は y）
          "minmax" 全点の外接矩形
    """
    if mode == "mean":
        lx = q[:, 0, :, 0].mean(axis=1)           # 左辺の x
        rx = q[:, 1, :, 0].mean(axis=1)           # 右辺の x
        ty = q[:, 2, :, 1].mean(axis=1)           # 上辺の y
        by = q[:, 3, :, 1].mean(axis=1)           # 下辺の y
    elif mode == "minmax":
        flat = q.reshape(q.shape[0], -1, 2)       # (K, 4n, 2)
        lx, ty = flat[:, :, 0].min(axis=1), flat[:, :, 1].min(axis=1)
        rx, by = flat[:, :, 0].max(axis=1), flat[:, :, 1].max(axis=1)
    else:
        raise ValueError(f"mode は {BOX_MODES} のどれか: {mode}")
    # 左右・上下が入れ替わっても正しく作る（極端な変形への保険）
    x1, x2 = np.minimum(lx, rx), np.maximum(lx, rx)
    y1, y2 = np.minimum(ty, by), np.maximum(ty, by)
    return np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)


def _transform_bbox(bboxes, point_fn, mode, n_per_side):
    """
    forward_bbox / inverse_bbox の共通部分。
    point_fn  : (K, 2) の点 -> (K, 2) の点 を返す関数（forward_points か inverse_points）
    mode      : BOX_MODES のどれか
    n_per_side: 1本の辺に置く点の数（2 以上。corner2 では使わない）
    """
    b = np.asarray(bboxes, dtype=float).reshape(-1, 4)
    if len(b) == 0:
        return b
    if mode == "corner2":
        return _corners_to_bbox(point_fn(_bbox_to_corners(b)))   # 従来と同じ計算
    if n_per_side < 2:
        raise ValueError("n_per_side は 2 以上（両端の角を含むため）")
    p = _side_points(b, n_per_side)               # (K, 4, n, 2)
    q = point_fn(p.reshape(-1, 2)).reshape(p.shape)
    return _aggregate_sides(q, mode)


def forward_bbox(bboxes, fmap, mode=DEFAULT_MODE, n_per_side=DEFAULT_N_PER_SIDE):
    """
    COCO ボックス (K, 4) を元画像 -> 変形後へ。
    mode, n_per_side はモジュール先頭の「ボックスの作り方」を参照
    """
    return _transform_bbox(bboxes, lambda p: forward_points(p, fmap), mode, n_per_side)


def inverse_bbox(bboxes, map_x, map_y, mode=DEFAULT_MODE, n_per_side=DEFAULT_N_PER_SIDE):
    """
    COCO ボックス (K, 4) を変形後 -> 元画像へ。
    mode, n_per_side は forward_bbox と同じものを使う（正解と検出で作り方をそろえるため）
    """
    return _transform_bbox(bboxes, lambda p: inverse_points(p, map_x, map_y), mode, n_per_side)


def _iou_xywh(a, b):
    """同じ並びのボックス同士の IoU。a, b: [x, y, w, h] (K, 4) -> (K,)"""
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(a[:, 0], b[:, 0]), 0, None)   # 重なりの幅
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(a[:, 1], b[:, 1]), 0, None)   # 重なりの高さ
    inter = iw * ih
    union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def roundtrip_iou(bboxes, fmap, map_x, map_y, mode="corner2", n_per_side=3):
    """
    元の正解 -> 変形後 -> 元画像 と往復させたボックスと、元の正解との IoU (K,) を返す。
    ボックスの作り方（mode）を比べる目安。1 に近いほど、往復で枠が崩れていない。
    注意: 順写像と逆写像は互いの近似なので、点1つでも完全には戻らない。
          corner2 の値を基準にして、他の mode との差を見る。
    """
    b = np.asarray(bboxes, dtype=float).reshape(-1, 4)
    fwd = forward_bbox(b, fmap, mode, n_per_side)              # 変形後の枠
    back = inverse_bbox(fwd, map_x, map_y, mode, n_per_side)   # 元画像へ戻した枠
    return _iou_xywh(b, back)


# ===========================================================================
# 確認用: 正解ボックスを画像に描いて保存する
# ===========================================================================
# 色を統一するときの色（BGR の赤）
SINGLE_COLOR = (0, 0, 255)
# カテゴリ id ごとの色（BGR）。VisDrone の 10 クラスを見分けやすいように変える
_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
           (255, 255, 0), (0, 128, 255), (255, 128, 0), (128, 0, 255), (0, 255, 128)]


def draw_gt(img, anns, thickness=1, color=SINGLE_COLOR):
    """
    画像に正解ボックスを描いた画像を返す（元の画像は変更しない）。
    anns : COCO 形式の正解のリスト（'bbox' = [x, y, w, h], 'category_id'）
    color: ボックスの色 (B, G, R)。None ならカテゴリごとに色を分ける
    ボックスの座標は画素の端基準なので、右下は x + w の1つ手前の画素まで描く。
    """
    out = img.copy()
    for a in anns:
        x, y, w, h = a["bbox"]
        c = color if color is not None else _COLORS[(a.get("category_id", 1) - 1) % len(_COLORS)]
        p1 = (int(round(x)), int(round(y)))
        p2 = (max(int(round(x + w)) - 1, p1[0]), max(int(round(y + h)) - 1, p1[1]))
        cv2.rectangle(out, p1, p2, c, thickness)
    return out


def save_gt_check(out_dir, name, orig_img, orig_anns, warped_img, warped_anns, color=SINGLE_COLOR):
    """
    color: ボックスの色。None ならカテゴリごとに色を分ける（draw_gt と同じ）
    3枚を保存する:
      1. {name}_orig_gt.png    元画像 + 元の正解
      2. {name}_warped_gt.png  非一様縮小後の画像 + 変換後の正解
      3. {name}_gt_compare.png 1 と 2 を横に並べたもの
         元画像の方が大きいので、高さを非一様縮小後の画像にそろえて縮小して並べる（表示用）
    """
    os.makedirs(out_dir, exist_ok=True)
    # 元画像は大きいので線を太めにする（縮小して並べたときにも見えるように）
    scale = warped_img.shape[0] / orig_img.shape[0]
    orig_vis = draw_gt(orig_img, orig_anns, thickness=max(1, int(round(1 / scale))), color=color)
    warped_vis = draw_gt(warped_img, warped_anns, thickness=1, color=color)
    cv2.imwrite(os.path.join(out_dir, f"{name}_orig_gt.png"), orig_vis)
    cv2.imwrite(os.path.join(out_dir, f"{name}_warped_gt.png"), warped_vis)

    orig_small = cv2.resize(orig_vis, (int(round(orig_img.shape[1] * scale)), warped_img.shape[0]),
                            interpolation=cv2.INTER_AREA)
    gap = np.full((warped_img.shape[0], 10, 3), 255, np.uint8)   # 2枚の間の白い区切り
    cv2.imwrite(os.path.join(out_dir, f"{name}_gt_compare.png"),
                np.hstack([orig_small, gap, warped_vis]))


def select_targets(orig_img_args, orig_images, limit):
    """
    --orig-img で指定されたものから、確認する画像を決める。
      フォルダ    : 正解 json に載っている画像のうち、そのフォルダにあるものを名前順に limit 枚
      画像ファイル: そのファイル（何枚でも指定可。limit は使わない）
    Returns: [(元画像のパス, 正解 json の画像情報), ...]
    """
    by_name = {im["file_name"]: im for im in orig_images}   # ファイル名 -> 画像情報
    targets = []
    for path in orig_img_args:
        if os.path.isdir(path):
            names = sorted(n for n in by_name if os.path.isfile(os.path.join(path, n)))
            targets += [(os.path.join(path, n), by_name[n]) for n in names[:limit]]
        else:
            name = os.path.basename(path)
            if name not in by_name:
                print(f"[スキップ] 正解 json に載っていない画像です: {path}")
                continue
            targets.append((path, by_name[name]))
    return targets


def main():
    parser = argparse.ArgumentParser(description="元画像と非一様縮小後の画像に正解を描いて確認する")
    parser.add_argument("--orig-ann", required=True, help="元の正解 json（例: my_val.json）")
    parser.add_argument("--orig-img", nargs="+", required=True,
                        help="元画像のフォルダ、または画像ファイル（複数可）。"
                             "ファイル名で正解 json の画像と対応させる")
    parser.add_argument("--warp-dir", required=True,
                        help="make_deformed_dataset.py の --out-dir（images/ と annotations.json がある所）")
    parser.add_argument("--limit", type=int, default=3, help="フォルダを指定したときに確認する枚数（名前順）")
    parser.add_argument("--color-mode", choices=["single", "class"], default="single",
                        help="single: 全部同じ色（既定） / class: カテゴリごとに色を分ける")
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "gt_check"),
                        help="保存先。既定はこのファイルと同じ場所の gt_check/")
    args = parser.parse_args()

    with open(args.orig_ann) as f:
        orig = json.load(f)
    with open(os.path.join(args.warp_dir, "annotations.json")) as f:
        warped = json.load(f)

    # 画像 id ごとに正解をまとめる（make_deformed_dataset.py は画像 id を変えない）
    orig_anns, warped_anns = defaultdict(list), defaultdict(list)
    for a in orig["annotations"]:
        orig_anns[a["image_id"]].append(a)
    for a in warped["annotations"]:
        warped_anns[a["image_id"]].append(a)
    warped_images = {im["id"]: im for im in warped["images"]}

    targets = select_targets(args.orig_img, orig["images"], args.limit)
    if not targets:
        raise FileNotFoundError(f"確認できる画像がありません: {args.orig_img}")
    color = SINGLE_COLOR if args.color_mode == "single" else None

    for orig_path, o in targets:
        w = warped_images.get(o["id"])
        if w is None:
            print(f"[スキップ] 非一様縮小のデータにない画像です（--limit で作った場合など）: {o['file_name']}")
            continue
        orig_img = cv2.imread(orig_path)
        warped_img = cv2.imread(os.path.join(args.warp_dir, "images", w["file_name"]))
        if orig_img is None or warped_img is None:
            print(f"[スキップ] 画像を読めません: {o['file_name']}")
            continue
        name = os.path.splitext(o["file_name"])[0]
        save_gt_check(args.out_dir, name, orig_img, orig_anns[o["id"]], warped_img, warped_anns[o["id"]],
                      color=color)
        print(f"[{name}] 元 {orig_img.shape[1]}x{orig_img.shape[0]} / 変形後 "
              f"{warped_img.shape[1]}x{warped_img.shape[0]}, 正解 {len(warped_anns[o['id']])} 個")
    print(f"保存先: {args.out_dir}")


if __name__ == "__main__":
    main()