"""元画像・非一様縮小後の画像に、正解と検出結果を描いて確認する。

box_transform.py の main（正解を描く確認）に、検出結果の描画を足したもの。
変形後の画像・変形後の正解・逆写像は、make_deformed_dataset.py で作成済みのもの（--warp-dir）を読む。
逆写像（meshes/*.npz）は --save-mesh で作ったデータ（val）にしかないので、val で使う。

1枚の画像につき、次を保存する:
  {name}_orig_gt.png       元画像 + 元の正解
  {name}_warped_gt.png     非一様縮小後の画像 + 変換後の正解
  {name}_orig_det.png      元画像 + 元の座標へ戻した検出結果
  {name}_warped_det.png    非一様縮小後の画像 + 検出結果（検出器が出したまま）
  {name}_compare.png       上の4枚を 2x2 に並べ、それぞれの上にタイトルを付けたもの
                           （左の列 = 元画像、右の列 = 非一様縮小後 / 上の行 = 正解、下の行 = 検出）

実行例（projects/Edge_poisson_asap/ で）:
  PYTHONPATH=. python tools/draw_detection.py \
      --orig-ann ../../data_ann/VisDrone/my_val.json \
      --orig-img ../../data/VisDrone/VisDrone2019-DET-val/images  \
      --warp-dir ../../data_ann/VisDrone_poisson_asap/val \
      --config configs/yolox_tiny_visdrone_poisson_asap.py --checkpoint ../../work_dirs/yolox_tiny_visdrone_poisson_asap/epoch_50.pth \
      --out-dir ../../work_dirs/eval_visdrone_poisson_asap \
      --limit 3

    PYTHONPATH=. python tools/draw_detection.py \
        --orig-ann ../../data_ann/VisDrone/my_val.json \
        --orig-img ../../data/VisDrone/VisDrone2019-DET-val/images/0000277_03401_d_0000555.jpg \
        --warp-dir ../../data_ann/VisDrone_poisson_asap/val \
        --config configs/yolox_tiny_visdrone_poisson_asap.py --checkpoint ../../work_dirs/yolox_tiny_visdrone_poisson_asap_e50/epoch_50.pth \
        --out-dir ../../work_dirs/eval_visdrone_poisson_asap_e50/detection_images
"""
import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np
from mmdet.apis import inference_detector, init_detector

from edge_poisson_asap.box_transform import SINGLE_COLOR, draw_gt, inverse_bbox, select_targets

DET_COLOR = (0, 255, 0)   # 検出の枠の色（BGR で緑）。正解（SINGLE_COLOR = 赤）と見分けるため
TITLE_H = 32              # 一覧画像で各図の上に付けるタイトル欄の高さ [px]
GAP = 10                  # 一覧画像の図どうしの白い区切りの幅 [px]


def to_anns(bboxes, category_ids):
    """ボックス (K, 4) とカテゴリ id を、draw_gt が受け取る COCO 形式の dict のリストにする"""
    return [{"bbox": list(map(float, b)), "category_id": int(c)} for b, c in zip(bboxes, category_ids)]


def load_inverse_map(warp_dir, stem):
    """make_deformed_dataset.py --save-mesh で保存した逆写像 (map_x, map_y) を読む。なければ None"""
    path = os.path.join(warp_dir, "meshes", stem + ".npz")
    if not os.path.isfile(path):
        return None
    with np.load(path) as z:
        return z["map_x"], z["map_y"]


def detect(model, img, score_thr, label_to_cat):
    """非一様縮小後の画像（BGR）で推論し、score_thr 以上の検出を返す。
    Returns: ボックス (K, 4) [x, y, w, h]（変形後の座標）, カテゴリ id (K,)"""
    inst = inference_detector(model, img).pred_instances   # 配列を渡すと読み込み処理が配列用になる
    keep = inst.scores >= score_thr
    xyxy = inst.bboxes[keep].cpu().numpy()
    xywh = np.c_[xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]]   # x1y1x2y2 -> COCO の xywh
    cats = [label_to_cat[int(l)] for l in inst.labels[keep].cpu().numpy()]
    return xywh, cats


def add_title(img, title):
    """画像の上に白いタイトル欄を付けて返す（cv2 は日本語を描けないので英語で書く）"""
    bar = np.full((TITLE_H, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def save_check(out_dir, name, orig_img, warped_img, orig_gts, warped_gts, orig_dets, warped_dets,
               gt_color=SINGLE_COLOR, det_color=DET_COLOR):
    """正解と検出を描いた4枚を個別に保存し、2x2 に並べた一覧（タイトル付き）も保存する。
    線の太さと元画像の縮小の仕方は box_transform.save_gt_check と同じにしている"""
    scale = warped_img.shape[0] / orig_img.shape[0]     # 元画像の高さを変形後の画像にそろえる倍率
    thick = max(1, int(round(1 / scale)))               # 元画像は大きいので線を太めにする
    panels = {   # 名前: (描いた画像, 一覧でのタイトル)
        "orig_gt": (draw_gt(orig_img, orig_gts, thickness=thick, color=gt_color),
                    f"Original + GT ({len(orig_gts)})"),
        "warped_gt": (draw_gt(warped_img, warped_gts, thickness=1, color=gt_color),
                      f"Warped + GT ({len(warped_gts)})"),
        "orig_det": (draw_gt(orig_img, orig_dets, thickness=thick, color=det_color),
                     f"Original + Det, mapped back ({len(orig_dets)})"),
        "warped_det": (draw_gt(warped_img, warped_dets, thickness=1, color=det_color),
                       f"Warped + Det ({len(warped_dets)})"),
    }
    # 4枚を原寸で個別に保存する（細かい確認用）
    for key, (img, _) in panels.items():
        cv2.imwrite(os.path.join(out_dir, f"{name}_{key}.png"), img)

    small_size = (int(round(orig_img.shape[1] * scale)), warped_img.shape[0])

    def cell(key):
        """一覧の1マス: 元画像は変形後の画像と同じ高さに縮小し、タイトルを付ける"""
        img, title = panels[key]
        if key.startswith("orig"):
            img = cv2.resize(img, small_size, interpolation=cv2.INTER_AREA)
        return add_title(img, title)

    vgap = np.full((TITLE_H + warped_img.shape[0], GAP, 3), 255, np.uint8)   # 左右の区切り
    top = np.hstack([cell("orig_gt"), vgap, cell("warped_gt")])          # 上の行 = 正解
    bottom = np.hstack([cell("orig_det"), vgap, cell("warped_det")])     # 下の行 = 検出
    hgap = np.full((GAP, top.shape[1], 3), 255, np.uint8)                # 上下の区切り
    cv2.imwrite(os.path.join(out_dir, f"{name}_compare.png"), np.vstack([top, hgap, bottom]))


def parse_args():
    parser = argparse.ArgumentParser(description="元画像と非一様縮小後の画像に、正解と検出結果を描いて確認する")
    parser.add_argument("--orig-ann", required=True, help="元の正解 json（例: my_val.json）")
    parser.add_argument("--orig-img", nargs="+", required=True,
                        help="元画像のフォルダ、または画像ファイル（複数可）")
    parser.add_argument("--warp-dir", required=True,
                        help="make_deformed_dataset.py の --out-dir（images/, annotations.json, meshes/ がある所）")
    parser.add_argument("--limit", type=int, default=3, help="フォルダを指定したときに確認する枚数（名前順）")
    parser.add_argument("--config", required=True, help="mmdet の設定ファイル")
    parser.add_argument("--checkpoint", required=True, help="学習済みの重み")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-thr", type=float, default=0.3, help="描く検出のスコアの下限")
    parser.add_argument("--color-mode", choices=["single", "class"], default="single",
                        help="single: 正解は赤・検出は緑で統一（既定） / class: カテゴリごとに色を分ける")
    parser.add_argument("--out-dir", default="work_dirs/det_check", help="保存先")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.orig_ann) as f:
        orig = json.load(f)
    with open(os.path.join(args.warp_dir, "annotations.json")) as f:
        warped = json.load(f)
    # 作成時の設定を表示しておく（どの条件のデータで確認したかを残すため）
    settings_path = os.path.join(args.warp_dir, "settings.json")
    if os.path.isfile(settings_path):
        with open(settings_path) as f:
            print(f"変形後のデータの作成時の設定: {json.load(f)}")

    # 画像 id ごとに正解をまとめる（make_deformed_dataset.py は画像 id を変えない）
    orig_anns, warped_anns = defaultdict(list), defaultdict(list)
    for a in orig["annotations"]:
        orig_anns[a["image_id"]].append(a)
    for a in warped["annotations"]:
        warped_anns[a["image_id"]].append(a)
    warped_images = {im["id"]: im for im in warped["images"]}

    model = init_detector(args.config, args.checkpoint, device=args.device)
    # 検出器のクラス番号 -> 正解 json のカテゴリ id（クラス名で対応させる）
    name_to_cat = {c["name"]: c["id"] for c in orig["categories"]}
    label_to_cat = [name_to_cat[n] for n in model.dataset_meta["classes"]]

    targets = select_targets(args.orig_img, orig["images"], args.limit)
    if not targets:
        raise FileNotFoundError(f"確認できる画像がありません: {args.orig_img}")
    gt_color = SINGLE_COLOR if args.color_mode == "single" else None
    det_color = DET_COLOR if args.color_mode == "single" else None
    os.makedirs(args.out_dir, exist_ok=True)

    for orig_path, o in targets:
        w = warped_images.get(o["id"])
        name = os.path.splitext(o["file_name"])[0]
        if w is None:
            print(f"[スキップ] 非一様縮小のデータにない画像です（--limit で作った場合など）: {o['file_name']}")
            continue
        maps = load_inverse_map(args.warp_dir, os.path.splitext(w["file_name"])[0])
        if maps is None:
            print(f"[スキップ] 逆写像がありません（--save-mesh なしで作ったデータ）: {name}")
            continue
        orig_img = cv2.imread(orig_path)
        warped_img = cv2.imread(os.path.join(args.warp_dir, "images", w["file_name"]))
        if orig_img is None or warped_img is None:
            print(f"[スキップ] 画像を読めません: {o['file_name']}")
            continue

        # 検出: 変形後の画像で推論し、保存済みの逆写像で元の座標へ戻す
        det_boxes, det_cats = detect(model, warped_img, args.score_thr, label_to_cat)
        orig_dets = to_anns(inverse_bbox(det_boxes, *maps), det_cats)

        # 正解（元の正解と作成済みの変形後の正解）と検出を描き、4枚と一覧を保存する
        save_check(args.out_dir, name, orig_img, warped_img,
                   orig_anns[o["id"]], warped_anns[o["id"]], orig_dets, to_anns(det_boxes, det_cats),
                   gt_color=gt_color, det_color=det_color)

        print(f"[{name}] 元 {orig_img.shape[1]}x{orig_img.shape[0]} / 変形後 "
              f"{warped_img.shape[1]}x{warped_img.shape[0]}, "
              f"正解 {len(warped_anns[o['id']])} 個, 検出 {len(det_cats)} 個")
    print(f"保存先: {args.out_dir}")


if __name__ == "__main__":
    main()