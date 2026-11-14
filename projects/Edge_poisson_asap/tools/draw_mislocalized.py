"""位置ずれの検出（クラスは正しいが IoU が低い検出）を、その場で推論して画像に描く。

流れ（draw_detection.py と同じく、検出結果の json は使わない）:
  1. 変形後の画像（{warp-dir}/images/）を検出器に通す
  2. 検出を、保存済みの逆写像（{warp-dir}/meshes/*.npz の map_x, map_y）で元の座標へ戻す
  3. mislocalized_cases.collect_cases で「クラスは正しいが元の座標での IoU が低い」検出を集める
  4. 1件ごとに4枚のパネルを作り、横に並べた図と一緒に保存する

1件につき4枚（ボックスは1枚に1つだけ描く。左から順に横に並べる）:
  1. 元画像の正解（緑）
  2. 変形後の正解（緑）                     {warp-dir}/annotations.json のもの
  3. 変形後の画像での検出（赤、検出器が出したまま）  タイトルに変形後の座標での IoU
  4. 元の座標へ戻した検出（赤）             タイトルに元の座標での IoU
同じ座標系の2枚（1と4、2と3）は同じ範囲を同じ倍率で切り出すので、並べると位置を比べられる。
横に並べた図の下には、番号・画像の名前・クラス・サイズ・スコア・2つの IoU・image_id・ann_id を書く。

保存するもの:
  {out-dir}/{元画像の名前}/{番号}_compare.png          4枚を横に並べた図
  {out-dir}/{元画像の名前}/{番号}_1_gt_orig.png など   4枚を1枚ずつ
  {out-dir}/cases.csv                                  図にした候補だけの一覧（番号順）
  {番号} はスコア順などで並べたときの順位（3桁）。1枚の元画像に複数の候補があっても重ならない。
  cases.csv の rank 列が図の番号、file_name 列が元画像の名前なので、図から行をそのまま探せる。
  （起動時に表示する集計は、図にしなかったものも含めた全件の数）

必要なもの:
  --warp-dir は make_deformed_dataset.py を --save-mesh 付きで実行して作ったデータ。
  npz に map_x, map_y が入っていない古いデータ（11/07 より前に作ったもの）では止まる。

実行例（projects/Edge_poisson_asap/ で）:
  PYTHONPATH=. python tools/draw_mislocalized.py \
      --orig-ann ../../data_ann/VisDrone/my_val.json \
      --orig-img ../../data/VisDrone/VisDrone2019-DET-val/images \
      --warp-dir ../../data_ann/VisDrone_poisson_asap_mesh/val \
      --config configs/yolox_tiny_visdrone_poisson_asap.py \
      --checkpoint ../../work_dirs/yolox_tiny_visdrone_poisson_asap_e50/epoch_50.pth \
      --out-dir ../../work_dirs/eval_visdrone_poisson_asap_e50/mislocalized --score-thr 0.3
"""
import argparse
import csv
import os

import cv2
import numpy as np
from mmdet.apis import inference_detector, init_detector

from edge_poisson_asap.box_transform import (DEFAULT_MODE, DEFAULT_N_PER_SIDE, inverse_bbox,
                                             select_targets)
from mislocalized_cases import collect_cases, load_json

GT_COLOR = (0, 255, 0)    # 正解の枠の色（BGR で緑）
DET_COLOR = (0, 0, 255)   # 検出の枠の色（BGR で赤）
TEXT_H = 22               # 文字欄の高さ [px]
GAP = 6                   # パネルどうしの隙間 [px]


# ===========================================================================
# 推論と元の座標への変換
# ===========================================================================
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


def detect(model, img, score_thr, label_to_cat):
    """変形後の画像（BGR）で推論し、score_thr 以上の検出を返す。
    （draw_detection.detect と同じ処理に、スコアも返すようにしたもの）
    Returns: ボックス (K, 4) [x, y, w, h]（変形後の座標）, スコア (K,), カテゴリ id (K,)"""
    inst = inference_detector(model, img).pred_instances   # 配列を渡すと読み込み処理が配列用になる
    keep = inst.scores >= score_thr
    xyxy = inst.bboxes[keep].cpu().numpy()
    xywh = np.c_[xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]]   # x1y1x2y2 -> COCO の xywh
    cats = [label_to_cat[int(l)] for l in inst.labels[keep].cpu().numpy()]
    return xywh, inst.scores[keep].cpu().numpy(), cats


def to_det_dicts(image_id, boxes, scores, cats):
    """検出を COCO の結果形式（results.bbox.json と同じ形）の dict のリストにする"""
    return [dict(image_id=image_id, category_id=int(c), bbox=[float(v) for v in b], score=float(s))
            for b, s, c in zip(boxes, scores, cats)]


def run_detection(model, targets, warped_images, warp_dir, score_thr, label_to_cat):
    """対象の画像ごとに、変形後の画像で推論し、逆写像で元の座標へ戻す。
    Returns: (元の座標の検出, 変形後の座標の検出)  どちらも results.bbox.json と同じ形で、同じ並び
    （同じ並びにしておくと、collect_cases の中で2つが1対1に対応付けられる）"""
    det_orig, det_def = [], []
    for n, (_, o) in enumerate(targets, 1):
        w = warped_images.get(o["id"])
        maps = load_inverse_map(warp_dir, os.path.splitext(w["file_name"])[0]) if w else None
        if maps is None:
            print(f"[スキップ] 変形後のデータか逆写像の npz がありません: {o['file_name']}")
            continue
        img_d = read_image(os.path.join(warp_dir, "images", w["file_name"]))
        boxes, scores, cats = detect(model, img_d, score_thr, label_to_cat)
        det_def += to_det_dicts(w["id"], boxes, scores, cats)
        # 逆変換の方式は box_transform.py の既定値（DEFAULT_MODE, DEFAULT_N_PER_SIDE）
        det_orig += to_det_dicts(o["id"], inverse_bbox(boxes, *maps), scores, cats)
        if n % 50 == 0 or n == len(targets):
            print(f"推論 {n}/{len(targets)} 枚")
    return det_orig, det_def


# ===========================================================================
# 描画
# ===========================================================================
def read_image(path):
    """画像を読み込む。見つからなければパスを示して止める"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img


def crop_region(boxes, img_w, img_h, margin_ratio=1.0, min_margin=16):
    """複数のボックス（COCO 形式）をすべて含む切り出し範囲 (x1, y1, x2, y2) を、余白をつけて返す"""
    b = np.asarray(boxes, dtype=np.float64)
    x1, y1 = b[:, 0].min(), b[:, 1].min()
    x2, y2 = (b[:, 0] + b[:, 2]).max(), (b[:, 1] + b[:, 3]).max()
    # 小物体でも周りが見えるよう、余白は「枠の大きさ×倍率」と最低値の大きい方にする
    m = max(min_margin, margin_ratio * max(x2 - x1, y2 - y1))
    return (int(max(0, x1 - m)), int(max(0, y1 - m)),
            int(min(img_w, np.ceil(x2 + m))), int(min(img_h, np.ceil(y2 + m))))


def shared_region(img, boxes):
    """同じ座標系の正解と検出をどちらも含む切り出し範囲を返す。画像かボックスがなければ None"""
    boxes = [b for b in boxes if b is not None]
    if img is None or not boxes:
        return None
    h, w = img.shape[:2]
    return crop_region(boxes, w, h)


def blank_panel(out_size, title):
    """上に文字欄のある白いパネルを返す（画像がないときもこれを使う）"""
    canvas = np.full((out_size + TEXT_H, out_size, 3), 255, np.uint8)
    cv2.putText(canvas, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def draw_panel(img, box, region, out_size, color, title):
    """画像を region (x1, y1, x2, y2) で切り出して拡大し、ボックスを1つだけ描いたパネルを返す"""
    if img is None or region is None:
        return blank_panel(out_size, title + " (no data)")
    x1, y1, x2, y2 = region
    crop = img[y1:y2, x1:x2]
    scale = out_size / max(crop.shape[:2])        # 長い辺を out_size にそろえる倍率
    new_w = min(out_size, max(1, round(crop.shape[1] * scale)))
    new_h = min(out_size, max(1, round(crop.shape[0] * scale)))
    # 画素の形が見えるよう、最近傍補間で拡大する
    crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    if box is not None:
        # ボックスは画素の端基準（COCO と同じ）。右下は x + w の1つ手前の画素まで描く
        # （box_transform.draw_gt と同じ約束）
        p1 = (int(round((box[0] - x1) * scale)), int(round((box[1] - y1) * scale)))
        p2 = (max(int(round((box[0] + box[2] - x1) * scale)) - 1, p1[0]),
              max(int(round((box[1] + box[3] - y1) * scale)) - 1, p1[1]))
        cv2.rectangle(crop, p1, p2, color, 2)
    canvas = blank_panel(out_size, title)
    canvas[TEXT_H:TEXT_H + new_h, :new_w] = crop
    return canvas


def make_footer(lines, width):
    """横に並べた図の下に付ける文字欄（1行ずつ）を作る"""
    footer = np.full((TEXT_H * len(lines), width, 3), 255, np.uint8)
    for k, line in enumerate(lines):
        cv2.putText(footer, line, (4, 16 + TEXT_H * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    cv2.LINE_AA)
    return footer


def make_case_image(case, img_o, img_d, cat_name, out_size, rank, file_name):
    """1件分の4枚のパネルと、それを横に並べた図を作る。
    Returns: ({パネル名: 画像}, 横に並べた図)"""
    # 同じ座標系の2枚は同じ範囲で切り出す
    reg_o = shared_region(img_o, [case["gt_bbox_orig"], case["det_bbox_orig"]])
    reg_d = shared_region(img_d, [case["gt_bbox_def"], case["det_bbox_def"]])
    panels = {
        "1_gt_orig": draw_panel(img_o, case["gt_bbox_orig"], reg_o, out_size, GT_COLOR,
                                "1 GT (original)"),
        "2_gt_deformed": draw_panel(img_d, case["gt_bbox_def"], reg_d, out_size, GT_COLOR,
                                    "2 GT (deformed)"),
        "3_det_deformed": draw_panel(img_d, case["det_bbox_def"], reg_d, out_size, DET_COLOR,
                                     f"3 det (deformed) IoU={case['iou_def']:.2f}"),
        "4_det_orig": draw_panel(img_o, case["det_bbox_orig"], reg_o, out_size, DET_COLOR,
                                 f"4 det (back to orig) IoU={case['iou_orig']:.2f}"),
    }
    # 横に 1 → 2 → 3 → 4 の順（処理の流れの順）に並べる
    vgap = np.full((out_size + TEXT_H, GAP, 3), 255, np.uint8)
    row = panels["1_gt_orig"]
    for key in ("2_gt_deformed", "3_det_deformed", "4_det_orig"):
        row = np.hstack([row, vgap, panels[key]])
    lines = [f"#{rank:03d}  {file_name}  image_id={case['image_id']}  ann_id={case['ann_id']}",
             f"class={cat_name}  size={case['size']}  score={case['score']:.3f}  "
             f"IoU(original)={case['iou_orig']:.3f}  IoU(deformed)={case['iou_def']:.3f}"]
    return panels, np.vstack([row, make_footer(lines, row.shape[1])])


def save_case(out_dir, image_stem, rank, panels, grid):
    """横に並べた図と4枚を {out_dir}/{元画像の名前}/ に保存する。
    ファイル名の先頭に番号を付けて、同じ画像の別の候補と重ならないようにする"""
    img_dir = os.path.join(out_dir, image_stem)
    os.makedirs(img_dir, exist_ok=True)
    cv2.imwrite(os.path.join(img_dir, f"{rank:03d}_compare.png"), grid)
    for pname, img in panels.items():
        cv2.imwrite(os.path.join(img_dir, f"{rank:03d}_{pname}.png"), img)


def select_cases(cases, args):
    """描画する検出を、サイズとグループで絞り込み、並べ替えて上位を返す"""
    sel = [c for c in cases if args.size == "all" or c["size"] == args.size]
    if args.group == "back":        # 変形後の座標では範囲の上限を超えている（戻すときに落ちた）
        sel = [c for c in sel if c["iou_def"] >= args.iou_max]
    elif args.group == "deformed":  # 変形後の座標でも上限未満（検出器の時点で合っていない）
        sel = [c for c in sel if c["iou_def"] < args.iou_max]
    # スコアの高い位置ずれほど AP への悪影響が大きいので、既定はスコアの高い順
    key = {"score": lambda c: -c["score"], "iou": lambda c: c["iou_orig"]}[args.sort]
    return sorted(sel, key=key)[:args.max_cases]


def add_rank(selected, orig_paths):
    """図にする候補に、図の番号（rank）と元画像の名前（file_name）を付ける（cases.csv と図で共通）"""
    for rank, c in enumerate(selected):
        c["rank"] = rank
        c["file_name"] = os.path.basename(orig_paths[c["image_id"]])
    return selected


def draw_cases(selected, args, gt_o, gt_d, orig_paths):
    """add_rank 済みの候補を描いて、{out-dir}/{元画像の名前}/ に保存する"""
    cat_names = {c["id"]: c["name"] for c in gt_o["categories"]}
    files_d = {im["id"]: im["file_name"] for im in gt_d["images"]}
    last_id, img_o, img_d = None, None, None     # 同じ画像を続けて読み込まないための記録
    for c in selected:
        if c["image_id"] != last_id:
            img_o = read_image(orig_paths[c["image_id"]])
            img_d = (read_image(os.path.join(args.warp_dir, "images", files_d[c["image_id_def"]]))
                     if c["image_id_def"] is not None else None)
            last_id = c["image_id"]
        panels, grid = make_case_image(c, img_o, img_d,
                                       cat_names.get(c["category_id"], str(c["category_id"])),
                                       args.panel_size, c["rank"], c["file_name"])
        save_case(args.out_dir, os.path.splitext(c["file_name"])[0], c["rank"], panels, grid)


# ===========================================================================
# 集計と書き出し
# ===========================================================================
def summarize(cases, iou_max):
    """サイズごとに「戻すときに範囲へ落ちた（変形後の座標では IoU>=上限）」と
    「変形後の座標でも上限未満」に分けて数え、表示する"""
    print("size    件数  戻すときに落ちた  変形後でも低い  不明")
    for size in ("all", "small", "medium", "large"):
        cs = [c for c in cases if size == "all" or c["size"] == size]
        back = sum(c["iou_def"] >= iou_max for c in cs)    # NaN は False になる
        unknown = sum(np.isnan(c["iou_def"]) for c in cs)
        print(f"{size:7s} {len(cs):5d}  {back:16d}  {len(cs) - back - unknown:14d}  {unknown:4d}")


def write_csv(selected, path):
    """図にした候補だけを、図の番号順に CSV に書き出す"""
    fields = ["rank", "file_name", "image_id", "image_id_def", "ann_id", "category_id", "size", "score",
              "iou_orig", "iou_def", "gt_bbox_orig", "det_bbox_orig", "gt_bbox_def", "det_bbox_def"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in selected:
            writer.writerow({k: c[k] for k in fields})


def print_settings(warp_dir):
    """変形後のデータの作成時の設定（settings.json）と、逆変換の方式を表示する"""
    path = os.path.join(warp_dir, "settings.json")
    if os.path.isfile(path):
        print(f"変形後のデータの作成時の設定: {load_json(path)}")
    print(f"検出を元の座標へ戻す方式: mode={DEFAULT_MODE}, n_per_side={DEFAULT_N_PER_SIDE}"
          "（box_transform.py の既定値）")


# ===========================================================================
# 引数と全体の流れ
# ===========================================================================
def parse_args():
    """コマンドライン引数を読む（画像・データ・検出器の指定は draw_detection.py と同じ）"""
    p = argparse.ArgumentParser(description="位置ずれの検出を、その場で推論して4枚のパネルに描く")
    p.add_argument("--orig-ann", required=True, help="元の正解 json（例: my_val.json）")
    p.add_argument("--orig-img", nargs="+", required=True,
                   help="元画像のフォルダ、または画像ファイル（複数可）")
    p.add_argument("--limit", type=int, default=None, help="フォルダを指定したときに処理する枚数（既定は全部）")
    p.add_argument("--warp-dir", required=True,
                   help="make_deformed_dataset.py --save-mesh の --out-dir（images/, annotations.json, meshes/）")
    p.add_argument("--config", required=True, help="mmdet の設定ファイル")
    p.add_argument("--checkpoint", required=True, help="学習済みの重み")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default="work_dirs/mislocalized_viz", help="保存先")
    p.add_argument("--score-thr", type=float, default=0.3, help="対象にする検出のスコアの下限")
    p.add_argument("--iou-min", type=float, default=0.0, help="元の座標の IoU の下限（以上）")
    p.add_argument("--iou-max", type=float, default=0.5, help="元の座標の IoU の上限（未満）")
    p.add_argument("--size", choices=["all", "small", "medium", "large"], default="all")
    p.add_argument("--group", choices=["all", "back", "deformed"], default="all",
                   help="back: 変形後の座標では IoU が上限以上 / deformed: 変形後の座標でも上限未満")
    p.add_argument("--sort", choices=["score", "iou"], default="score", help="描く順（番号の付け方）")
    p.add_argument("--max-cases", type=int, default=100, help="保存する候補の数")
    p.add_argument("--panel-size", type=int, default=320, help="1パネルの一辺 [px]")
    return p.parse_args()


def main():
    args = parse_args()
    print_settings(args.warp_dir)
    gt_o = load_json(args.orig_ann)
    gt_d = load_json(os.path.join(args.warp_dir, "annotations.json"))   # 変形後の正解
    warped_images = {im["id"]: im for im in gt_d["images"]}   # make_deformed_dataset.py は画像 id を変えない
    targets = select_targets(args.orig_img, gt_o["images"], args.limit)
    if not targets:
        raise FileNotFoundError(f"確認できる画像がありません: {args.orig_img}")

    # 検出器を作り、クラス番号 -> 正解 json のカテゴリ id の対応を作る（draw_detection.py と同じ）
    model = init_detector(args.config, args.checkpoint, device=args.device)
    name_to_cat = {c["name"]: c["id"] for c in gt_o["categories"]}
    label_to_cat = [name_to_cat[n] for n in model.dataset_meta["classes"]]
    det_orig, det_def = run_detection(model, targets, warped_images, args.warp_dir,
                                      args.score_thr, label_to_cat)

    # クラスは正しいが元の座標での IoU が低い検出を集める
    cases, n_fallback = collect_cases(gt_o, gt_d, det_orig, det_def,
                                      args.score_thr, args.iou_min, args.iou_max)
    print(f"対象の検出数: {len(cases)}（score>={args.score_thr}, {args.iou_min}<=IoU<{args.iou_max}）")
    print(f"並び順で対応付けられなかった画像の数: {n_fallback}（0 になるはず）")
    summarize(cases, args.iou_max)

    # 図にする候補を選んで番号を付け、同じものを cases.csv と図に出す
    orig_paths = {o["id"]: path for path, o in targets}
    selected = add_rank(select_cases(cases, args), orig_paths)
    print(f"図にする候補: {len(selected)} 件（--max-cases {args.max_cases}）")
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(selected, os.path.join(args.out_dir, "cases.csv"))
    draw_cases(selected, args, gt_o, gt_d, orig_paths)
    print(f"保存先: {args.out_dir}")


if __name__ == "__main__":
    main()