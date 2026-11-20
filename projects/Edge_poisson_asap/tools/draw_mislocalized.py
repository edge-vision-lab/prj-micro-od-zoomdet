"""位置ずれの検出（クラスは正しいが IoU が低い検出）を、その場で推論して画像に描く。

流れ（draw_detection.py と同じく、検出結果の json は使わない）:
  1. 変形後の画像（{warp-dir}/images/）を検出器に通す（画像はパスで渡す。tools/test.py と同じ読み込み処理）
  2. 検出を、保存済みの逆写像（{warp-dir}/meshes/*.npz の map_x, map_y）で元の座標へ戻す
  3. mislocalized_cases.collect_cases で「クラスは正しいが元の座標での IoU が低い」検出を集める
     （正解への割り当てをどちらの座標で行うかは --match-space で選ぶ）
  4. 変形後の座標での IoU の範囲（--iou-def-min, --iou-def-max）などで絞り込み、並べ替える
  5. 1件ごとに4枚のパネルを作り、横に並べた図と一緒に保存する
  2 で元の座標へ戻すときのボックスの作り方は --bbox-mode, --bbox-n-per-side で指定する
  （省略すると --warp-dir の settings.json に残っている、正解を作ったときの方式）
  1〜2 の推論と逆写像の読み込みは deformed_predict.py（ほかのツールと共通）

対象にする検出の条件（すべて AND）:
  元の座標の IoU    : --iou-min <= IoU < --iou-max            （既定 0.0 <= IoU < 0.5）
  変形後の座標の IoU: --iou-def-min <= IoU < --iou-def-max    （既定は条件なし）
  正解への割り当て  : --match-space orig（既定。元の座標で割り当てる。従来どおり）
                      --match-space deformed（変形後の座標で割り当て、元の座標では同じ正解との IoU を測る）
  「変形後の座標では合っていたのに、元の座標へ戻すと崩れた（裏返りなど）」検出を見るときは
  --match-space deformed を使う。orig のままだと、元の座標で IoU=0 まで崩れた検出は「相手なし」として
  集める前に落ち、隣の別の正解に割り当てられた検出は別の正解との IoU になってしまう。

1件につき4枚（ボックスは1枚に1つだけ描く。左から順に横に並べる）:
  1. 元画像の正解（緑）
  2. 変形後の正解（緑）                     {warp-dir}/annotations.json のもの
  3. 変形後の画像での検出（赤、検出器が出したまま）  タイトルに変形後の座標での IoU
  4. 元の座標へ戻した検出（赤）             タイトルに元の座標での IoU
同じ座標系の2枚（1と4、2と3）は同じ範囲を同じ倍率で切り出すので、並べると位置を比べられる。
横に並べた図の下には、番号・画像の名前・クラス・サイズ・スコア・2つの IoU・image_id・ann_id を書く。

保存するもの（{保存先} = {out-dir}_{match-space}。例: mislocalized_orig, mislocalized_deformed）:
  {保存先}/{元画像の名前}/{番号}_compare.png          4枚を横に並べた図
  {保存先}/{元画像の名前}/{番号}_1_gt_orig.png など   4枚を1枚ずつ
  {保存先}/cases.csv                                  図にした候補だけの一覧（番号順）
  {保存先}/summary.txt                                件数の集計と実行時の設定（標準出力にも同じものを出す）
  割り当ての座標（--match-space）ごとにフォルダーを分けるので、2通りで実行しても上書きされない。
  --out-dir の名前がすでに _orig / _deformed で終わっていれば、重ねて付けない。
  {番号} はスコア順などで並べたときの順位（3桁）。1枚の元画像に複数の候補があっても重ならない。
  cases.csv の rank 列が図の番号、file_name 列が元画像の名前なので、図から行をそのまま探せる。
  （集計は、図にしなかったものも含めた全件の数）

件数の集計（summary.txt）:
  クラス正解で割り当てられた検出のうち、元の座標の IoU が --iou-min 以上 --iou-max 未満のものの数と、
  その内訳（変形後の座標の IoU が --summary-def-bins で区切った区間のどこか。既定は
  >=0.75 / 0.50-0.75 / <0.50 / 不明）を、サイズ別・クラス別に書く。
  ほかに、実行日時・コマンド・設定・画像と検出の数・もう一方の --match-space での件数（参考）・図にした件数を書く。

必要なもの:
  --warp-dir は make_deformed_dataset.py を --save-mesh 付きで実行して作ったデータ。
  npz に map_x, map_y が入っていない古いデータ（11/07 より前に作ったもの）では止まる。

実行例（projects/Edge_poisson_asap/ で）:
  PYTHONPATH=. python tools/draw_mislocalized.py \
      --orig-ann ../../data_ann/VisDrone/my_val.json \
      --orig-img ../../data/VisDrone/VisDrone2019-DET-val/images \
      --warp-dir ../../data_ann/VisDrone_poisson_asap_new/val \
      --config configs/yolox_tiny_visdrone_poisson_asap.py \
      --checkpoint ../../work_dirs/yolox_tiny_visdrone_poisson_asap_e50/epoch_50.pth \
      --out-dir ../../work_dirs/eval_visdrone_poisson_asap_e50/mislocalized --score-thr 0.3
  ボックスの変形方式を変えて比べる例: 上に --bbox-mode minmax --bbox-n-per-side all を付ける
  変形後の座標では IoU>=0.75 なのに、元の座標では IoU<0.5 になった検出を、IoU の落ち幅が大きい順に見る例:
    上に --match-space deformed --iou-def-min 0.75 --iou-max 0.5 --sort drop を付ける
"""
import argparse
import csv
import datetime
import os
import sys

import cv2
import numpy as np

from deformed_predict import label_to_cat_from_model, load_model, print_settings, run_detection
from edge_poisson_asap.box_method import add_box_args
from edge_poisson_asap.box_transform import select_targets
from eval_common import load_json
from mislocalized_cases import (MATCH_SPACES, collect_cases, def_bin_index, def_bin_labels,
                                filter_orig_range, summary_lines)

GT_COLOR = (0, 255, 0)    # 正解の枠の色（BGR で緑）
DET_COLOR = (0, 0, 255)   # 検出の枠の色（BGR で赤）
TEXT_H = 22               # 文字欄の高さ [px]
GAP = 6                   # パネルどうしの隙間 [px]


# ===========================================================================
# 推論する画像の選び方（推論そのものは deformed_predict.run_detection）
# ===========================================================================
def select_warped_images(targets, warped_images):
    """対象の元画像ごとに、推論する変形後の画像の情報を選ぶ。
    変形後のデータにない画像は飛ばし、元画像の名前を記録する（集計に枚数と名前を書くため）
    Returns: (変形後の画像の情報のリスト, 飛ばした元画像の名前のリスト,
              変形後の画像の名前 -> 元画像の名前 の辞書)"""
    images, skipped, to_orig_name = [], [], {}
    for _, o in targets:
        w = warped_images.get(o["id"])                 # make_deformed_dataset.py は画像 id を変えない
        if w is None:
            print(f"[スキップ] 変形後のデータにない画像です: {o['file_name']}")
            skipped.append(o["file_name"])
            continue
        images.append(w)
        to_orig_name[w["file_name"]] = o["file_name"]
    return images, skipped, to_orig_name


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


def make_case_image(case, img_o, img_d, cat_name, out_size, rank, file_name, match_space="orig"):
    """1件分の4枚のパネルと、それを横に並べた図を作る。
    match_space: 正解への割り当てをどちらの座標で行ったか（図の下の文字欄に書くだけ）
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
    lines = [f"#{rank:03d}  {file_name}  image_id={case['image_id']}  ann_id={case['ann_id']}"
             f"  match={match_space}",
             f"class={cat_name}  size={case['size']}  score={case['score']:.3f}  "
             f"IoU(original)={case['iou_orig']:.3f}  IoU(deformed)={case['iou_def']:.3f}"]
    return panels, np.vstack([row, make_footer(lines, row.shape[1])])


def save_case(out_dir, image_stem, rank, panels, grid):
    """横に並べた図と4枚を {out_dir}/{元画像の名前}/ に保存する（out_dir はモード名を付けた後の保存先）。
    ファイル名の先頭に番号を付けて、同じ画像の別の候補と重ならないようにする"""
    img_dir = os.path.join(out_dir, image_stem)
    os.makedirs(img_dir, exist_ok=True)
    cv2.imwrite(os.path.join(img_dir, f"{rank:03d}_compare.png"), grid)
    for pname, img in panels.items():
        cv2.imwrite(os.path.join(img_dir, f"{rank:03d}_{pname}.png"), img)


def in_def_range(case, iou_def_min, iou_def_max):
    """変形後の座標での IoU が iou_def_min 以上 iou_def_max 未満か（None の側は条件なし）。
    条件があるときに IoU が不明（NaN）なら、範囲に入るか判断できないので False にする"""
    if iou_def_min is None and iou_def_max is None:
        return True
    v = case["iou_def"]
    if np.isnan(v):
        return False
    return (iou_def_min is None or v >= iou_def_min) and (iou_def_max is None or v < iou_def_max)


def select_cases(cases, args):
    """描画する検出を、サイズ・グループ・変形後の座標での IoU の範囲で絞り込み、並べ替えて返す。
    上位 --max-cases 件に絞るのは呼び出し側で行う（条件に合う件数を集計に書くため）"""
    sel = [c for c in cases if args.size == "all" or c["size"] == args.size]
    if args.group == "back":        # 変形後の座標では範囲の上限を超えている（戻すときに落ちた）
        sel = [c for c in sel if c["iou_def"] >= args.iou_max]
    elif args.group == "deformed":  # 変形後の座標でも上限未満（検出器の時点で合っていない）
        sel = [c for c in sel if c["iou_def"] < args.iou_max]
    # 変形後の座標での IoU の範囲（例: 0.75 以上 = 変形後では合っていた）
    sel = [c for c in sel if in_def_range(c, args.iou_def_min, args.iou_def_max)]
    # スコアの高い位置ずれほど AP への悪影響が大きいので、既定はスコアの高い順。
    # drop は「変形後の IoU − 元の IoU」の大きい順（戻すときに崩れた度合いの大きい順。NaN は最後）
    key = {"score": lambda c: -c["score"],
           "iou": lambda c: c["iou_orig"],
           "drop": lambda c: (np.isnan(c["iou_def"]), -(c["iou_def"] - c["iou_orig"]))}[args.sort]
    return sorted(sel, key=key)


def add_rank(selected, orig_paths):
    """図にする候補に、図の番号（rank）と元画像の名前（file_name）を付ける（cases.csv と図で共通）"""
    for rank, c in enumerate(selected):
        c["rank"] = rank
        c["file_name"] = os.path.basename(orig_paths[c["image_id"]])
    return selected


def draw_cases(selected, args, gt_o, gt_d, orig_paths):
    """add_rank 済みの候補を描いて、{保存先}/{元画像の名前}/ に保存する"""
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
                                       args.panel_size, c["rank"], c["file_name"], args.match_space)
        save_case(args.out_dir, os.path.splitext(c["file_name"])[0], c["rank"], panels, grid)


# ===========================================================================
# 集計と書き出し
# ===========================================================================
def summarize(matched, cases, args, cat_names):
    """件数の集計を、表示・保存用の文字列の行のリストにして返す（表は mislocalized_cases.summary_lines）。
    matched: クラス正解で割り当てられた検出すべて / cases: そのうち元の座標の IoU が対象範囲のもの"""
    lines = summary_lines(matched, cases, args.summary_def_bins, args.iou_min, args.iou_max, cat_names)
    if args.iou_def_min is not None or args.iou_def_max is not None:
        # 図にする条件として指定した、変形後の座標の IoU の範囲に入る件数
        n = sum(in_def_range(c, args.iou_def_min, args.iou_def_max) for c in cases)
        lines.append(f"変形後の座標の IoU が {fmt_range(args.iou_def_min, args.iou_def_max)} のもの: {n} 件")
    return lines


def fmt_range(lo, hi):
    """範囲を「0.75 <= IoU < -」の形の文字列にする（None の側は「-」）"""
    return f"{lo if lo is not None else '-'} <= IoU < {hi if hi is not None else '-'}"


def report_header(args, box_mode, n_per_side, targets, skipped, det_orig):
    """集計の最初に書く、実行日時・コマンド・設定・画像と検出の数の行を返す"""
    settings_path = os.path.join(args.warp_dir, "settings.json")
    settings = load_json(settings_path) if os.path.isfile(settings_path) else "なし"
    n_img = len(targets) - len(skipped)                # 推論した画像の数
    n_det = sum(d["score"] >= args.score_thr for d in det_orig)
    return ["=== 位置ずれの検出の集計（draw_mislocalized.py） ===",
            f"実行日時: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
            f"コマンド: python {' '.join(sys.argv)}",
            f"元の正解: {args.orig_ann}",
            f"変形後のデータ: {args.warp_dir}（作成時の設定: {settings}）",
            f"検出器: {args.config} / {args.checkpoint}",
            f"逆変換のボックスの作り方: mode={box_mode}, n_per_side={n_per_side}",
            f"割り当ての座標: {args.match_space} / スコアの下限: {args.score_thr} / "
            f"元の座標の IoU の範囲: {args.iou_min} <= IoU < {args.iou_max}",
            f"画像: 対象 {len(targets)} 枚、推論 {n_img} 枚、スキップ {len(skipped)} 枚"
            + (f"（{', '.join(skipped[:5])}{' ほか' if len(skipped) > 5 else ''}）" if skipped else ""),
            f"検出（score>={args.score_thr}）: {n_det} 個（1枚あたり {n_det / max(n_img, 1):.1f} 個）",
            ""]


def other_space_lines(gt_o, gt_d, det_orig, det_def, args):
    """もう一方の --match-space で割り当てたときの件数（参考）の行を返す。
    推論はやり直さず、同じ検出で割り当てだけをやり直す"""
    other = [m for m in MATCH_SPACES if m != args.match_space][0]
    matched, _ = collect_cases(gt_o, gt_d, det_orig, det_def, args.score_thr, 0.0, float("inf"), other)
    cases = filter_orig_range(matched, args.iou_min, args.iou_max)
    top = sum(def_bin_index(c["iou_def"], args.summary_def_bins) == 0 for c in cases)
    return ["", f"[参考] --match-space {other} で割り当てた場合: 割り当て数 {len(matched)}、"
                f"元IoU<{args.iou_max:.2f} {len(cases)} 件、"
                f"うち変形後{def_bin_labels(args.summary_def_bins)[0]} {top} 件"]


def selection_lines(args, n_ranked, n_selected):
    """図にした候補の条件と件数の行を返す"""
    return ["", f"[図にした候補] size={args.size}, group={args.group}, "
                f"変形後の IoU: {fmt_range(args.iou_def_min, args.iou_def_max)}, 並び順: {args.sort}",
            f"  条件に合う {n_ranked} 件のうち、上位 {n_selected} 件を図にした（--max-cases {args.max_cases}）",
            f"  保存先: {args.out_dir}"]


def emit_report(lines, path):
    """集計の行を標準出力に出し、同じものを path に保存する"""
    text = "\n".join(lines) + "\n"
    print(text, end="")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"集計の保存先: {path}")


def write_csv(selected, path):
    """図にした候補だけを、図の番号順に CSV に書き出す"""
    fields = ["rank", "file_name", "image_id", "image_id_def", "ann_id", "category_id", "size", "score",
              "iou_orig", "iou_def", "gt_bbox_orig", "det_bbox_orig", "gt_bbox_def", "det_bbox_def"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in selected:
            writer.writerow({k: c[k] for k in fields})


# ===========================================================================
# 引数と全体の流れ
# ===========================================================================
def add_mode_suffix(out_dir, match_space):
    """保存先のフォルダー名の末尾に、割り当ての座標のモード名（_orig / _deformed）を付けて返す。
    2通りの割り当てで実行しても、結果が同じフォルダーに上書きされないようにするため。
    すでにどちらかのモード名で終わっていれば、指定したモードの名前に置き換える（二重に付けない）"""
    path = os.path.normpath(out_dir)                  # 末尾の「/」を取り除く
    for m in MATCH_SPACES:
        if path.endswith("_" + m):
            path = path[:-len("_" + m)]               # 付いているモード名をいったん外す
            break
    return f"{path}_{match_space}"


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
    p.add_argument("--out-dir", default="work_dirs/mislocalized_viz",
                   help="保存先。実際には末尾に _{match-space} を付けたフォルダーに保存する"
                        "（例: work_dirs/mislocalized_viz_orig）")
    p.add_argument("--score-thr", type=float, default=0.3, help="対象にする検出のスコアの下限")
    p.add_argument("--iou-min", type=float, default=0.0, help="元の座標の IoU の下限（以上）")
    p.add_argument("--iou-max", type=float, default=0.5, help="元の座標の IoU の上限（未満）")
    p.add_argument("--iou-def-min", type=float, default=None,
                   help="変形後の座標の IoU の下限（以上）。省略すると条件なし（例: 0.75）")
    p.add_argument("--iou-def-max", type=float, default=None,
                   help="変形後の座標の IoU の上限（未満）。省略すると条件なし")
    p.add_argument("--match-space", choices=MATCH_SPACES, default="orig",
                   help="正解への割り当てを行う座標。orig: 元の座標（既定・従来どおり） / "
                        "deformed: 変形後の座標（戻すときに崩れた検出を見るとき）")
    p.add_argument("--size", choices=["all", "small", "medium", "large"], default="all")
    p.add_argument("--group", choices=["all", "back", "deformed"], default="all",
                   help="back: 変形後の座標では IoU が上限以上 / deformed: 変形後の座標でも上限未満"
                        "（--iou-def-min, --iou-def-max と同時に指定すると両方の条件を満たすもの）")
    p.add_argument("--sort", choices=["score", "iou", "drop"], default="score",
                   help="描く順（番号の付け方）。score: スコアの高い順 / iou: 元の IoU の低い順 / "
                        "drop: 変形後の IoU − 元の IoU の大きい順")
    p.add_argument("--max-cases", type=int, default=100, help="保存する候補の数")
    p.add_argument("--summary-def-bins", type=float, nargs="+", default=[0.5, 0.75],
                   help="集計で、変形後の座標の IoU を区切る境目（小さい順）。"
                        "既定 0.5 0.75 -> >=0.75 / 0.50-0.75 / <0.50 / 不明")
    p.add_argument("--panel-size", type=int, default=320, help="1パネルの一辺 [px]")
    # 検出を元の座標へ戻すときのボックスの作り方。省略すると --warp-dir の settings.json の値
    add_box_args(p, from_settings=True)
    args = p.parse_args()
    # 変形後の IoU の範囲が空にならないか確かめる（打ち間違いで 0 件になるのを防ぐ）
    if args.iou_def_min is not None and args.iou_def_max is not None and args.iou_def_min >= args.iou_def_max:
        p.error(f"--iou-def-min ({args.iou_def_min}) は --iou-def-max ({args.iou_def_max}) より小さくしてください")
    # 集計の区切りは小さい順で、同じ値がないこと（区間が空になるのを防ぐ）
    if any(a >= b for a, b in zip(args.summary_def_bins, args.summary_def_bins[1:])):
        p.error(f"--summary-def-bins は小さい順に、同じ値を含めずに指定してください: {args.summary_def_bins}")
    # 保存先のフォルダー名に割り当てのモードを付ける（以降の args.out_dir はすべてこの名前）
    args.out_dir = add_mode_suffix(args.out_dir, args.match_space)
    return args


def main():
    args = parse_args()
    box_mode, n_per_side = print_settings(args.warp_dir, args.bbox_mode, args.bbox_n_per_side)
    gt_o = load_json(args.orig_ann)
    gt_d = load_json(os.path.join(args.warp_dir, "annotations.json"))   # 変形後の正解
    warped_images = {im["id"]: im for im in gt_d["images"]}   # make_deformed_dataset.py は画像 id を変えない
    targets = select_targets(args.orig_img, gt_o["images"], args.limit)
    if not targets:
        raise FileNotFoundError(f"確認できる画像がありません: {args.orig_img}")

    # 検出器を作り、クラス番号 -> 正解 json のカテゴリ id の対応を作る（draw_detection.py と同じ）
    model = load_model(args.config, args.checkpoint, device=args.device)   # score_thr は設定のまま
    label_to_cat = label_to_cat_from_model(model, gt_o["categories"])
    # 変形後の画像で推論し、逆写像で元の座標へ戻す（逆写像の npz がない画像は飛ばす）
    images, skipped, to_orig_name = select_warped_images(targets, warped_images)
    det_orig, det_def, skipped_mesh = run_detection(model, images, args.warp_dir, label_to_cat,
                                                    (box_mode, n_per_side), args.score_thr,
                                                    skip_missing=True)
    skipped += [to_orig_name[name] for name in skipped_mesh]   # 集計には元画像の名前で書く

    # クラスは正しく正解に割り当てられた検出をすべて集め（集計の割合の分母）、
    # そのうち元の座標の IoU が対象の範囲のもの（位置ずれ）に絞る。
    # （--match-space deformed なら、変形後の座標で割り当てて、元の座標では同じ正解との IoU を測る）
    matched, n_fallback = collect_cases(gt_o, gt_d, det_orig, det_def,
                                        args.score_thr, 0.0, float("inf"), args.match_space)
    cases = filter_orig_range(matched, args.iou_min, args.iou_max)
    cat_names = {c["id"]: c["name"] for c in gt_o["categories"]}

    # 図にする候補を選んで番号を付け、同じものを cases.csv と図に出す
    orig_paths = {o["id"]: path for path, o in targets}
    ranked = select_cases(cases, args)                 # 条件に合うもの（並べ替え済み）
    selected = add_rank(ranked[:args.max_cases], orig_paths)
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(selected, os.path.join(args.out_dir, "cases.csv"))

    # 件数の集計を標準出力と summary.txt に出す（図を描く前に保存して、途中で止まっても残るようにする）
    lines = report_header(args, box_mode, n_per_side, targets, skipped, det_orig)
    lines.append(f"並び順で対応付けられなかった画像の数: {n_fallback}（0 になるはず）")
    lines += [""] + summarize(matched, cases, args, cat_names)
    lines += other_space_lines(gt_o, gt_d, det_orig, det_def, args)
    lines += selection_lines(args, len(ranked), len(selected))
    emit_report(lines, os.path.join(args.out_dir, "summary.txt"))

    draw_cases(selected, args, gt_o, gt_d, orig_paths)
    print(f"保存先: {args.out_dir}")


if __name__ == "__main__":
    main()