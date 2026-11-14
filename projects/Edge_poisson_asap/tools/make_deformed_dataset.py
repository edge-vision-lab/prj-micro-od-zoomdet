"""
非一様縮小（ポアソン＋ASAP）したデータセットを事前に作るツール

やること（1枚ごと）
  1. 元画像を読み、get_expand_region でマップを作る（make_expand_region.py）
  2. nonuniform_resize で非一様縮小した画像（内容サイズ）を作り、PNG で保存する
     （JPEG で保存し直すと圧縮の劣化が加わり、一様縮小と条件がずれるので可逆の PNG にする）
  3. 正解ボックスを順写像で変形後の座標に変換する（左上と右下の2点を変換）
  4. --save-mesh を付けたときは、変形後のノード位置 dst と逆写像 map_x, map_y を保存する
     （評価で検出結果を元画像の座標に戻すときに使う。val だけでよい。
      逆写像も保存しておくと評価のたびに計算しなくて済む。val 548 枚で 1〜2GB 程度の見込み）
最後に、元の json と同じ形式で、画像の大きさとボックスを書き換えた json を保存する。

置き場所: projects/Edge_poisson_asap/tools/make_deformed_dataset.py
使い方（projects/Edge_poisson_asap/ で実行）:
  python tools/make_deformed_dataset.py \
      --ann ../../data_ann/VisDrone/my_val.json \
      --img-dir ../../data/VisDrone/VisDrone2019-DET-val/images \
      --out-dir ../../data_ann/VisDrone_poisson_asap/val \
      --save-mesh
  出力:
    {out-dir}/images/{元の名前}.png    非一様縮小した画像
    {out-dir}/annotations.json         変形後の座標の正解
    {out-dir}/meshes/{元の名前}.npz     変形後のノード位置と逆写像（--save-mesh のとき）
    {out-dir}/settings.json            作成時の設定（あとで条件を確認するため）
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import cv2
import numpy as np

# tools/ から実行しても edge_poisson_asap パッケージを読み込めるよう、
# 1つ上（projects/Edge_poisson_asap/）を import の探索先に加える
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge_poisson_asap.make_expand_region import get_expand_region   # noqa: E402
from edge_poisson_asap.poisson_asap_image import nonuniform_resize    # noqa: E402
from edge_poisson_asap.box_transform import forward_bbox               # noqa: E402

# 各ワーカー（並列処理のプロセス）が共有する設定。initializer で一度だけ入れる
_CFG = None


def _init_worker(cfg):
    global _CFG
    _CFG = cfg


def process_one(job):
    """
    1枚分の処理。job = (画像情報 dict, その画像の正解のリスト)
    Returns: (新しい画像情報, 新しい正解のリスト, ログ用の情報)
    """
    img_info, anns = job
    cfg = _CFG
    stem = os.path.splitext(img_info["file_name"])[0]
    img = cv2.imread(os.path.join(cfg["img_dir"], img_info["file_name"]))   # BGR, uint8
    if img is None:
        raise FileNotFoundError(f"画像を読めません: {img_info['file_name']}")

    # 1. マップ作成（確認用の画像は保存しない）
    region_map = get_expand_region(img, stem, show_img=False)
    # 2. 非一様縮小
    warped, fmap, (map_x, map_y) = nonuniform_resize(
        img, region_map, tuple(cfg["box"]), thresh=cfg["thresh"],
        scale_expand=cfg["scale_expand"], scale_shrink=cfg["scale_shrink"],
        w_target=cfg["w_target"], w_asap=cfg["w_asap"])
    cv2.imwrite(os.path.join(cfg["out_dir"], "images", stem + ".png"), warped)

    # 3. 正解ボックスを変形後の座標へ
    new_anns = []
    if anns:
        new_boxes = forward_bbox([a["bbox"] for a in anns], fmap)
        for a, b in zip(anns, new_boxes):
            na = dict(a)                                   # 他の項目（ignore など）はそのまま
            na["bbox"] = [round(float(v), 2) for v in b]
            na["area"] = round(float(b[2] * b[3]), 2)      # 学習時の判定用。評価は元画像の座標で行う
            new_anns.append(na)

    # 4. 評価用にノード位置と逆写像を保存（float32 で十分な精度。圧縮して保存）
    if cfg["save_mesh"]:
        np.savez_compressed(os.path.join(cfg["out_dir"], "meshes", stem + ".npz"),
                            dst=fmap.dst.astype(np.float32),
                            map_x=map_x.astype(np.float32), map_y=map_y.astype(np.float32),
                            in_size=np.array(fmap.in_size), content_size=np.array(fmap.content_size))

    new_info = dict(img_info)
    new_info["file_name"] = stem + ".png"
    new_info["height"], new_info["width"] = warped.shape[:2]
    # 元画像の大きさも残しておく（評価で元の座標に戻すときに使う）
    new_info["orig_height"], new_info["orig_width"] = img.shape[:2]
    log = {"region": float(region_map.mean()), "n_fold": fmap.n_fold,
           "n_small": int(sum(1 for a in new_anns if a["bbox"][2] < 1 or a["bbox"][3] < 1))}
    return new_info, new_anns, log


def main():
    parser = argparse.ArgumentParser(description="非一様縮小したデータセットを事前に作る")
    parser.add_argument("--ann", required=True, help="元の COCO 形式の json")
    parser.add_argument("--img-dir", required=True, help="元画像のフォルダ")
    parser.add_argument("--out-dir", required=True, help="保存先（元データとは別の書き込める場所）")
    parser.add_argument("--box", type=int, nargs=2, default=[800, 800], metavar=("W", "H"))
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--scale-expand", type=float, default=2.0)
    parser.add_argument("--scale-shrink", type=float, default=0.5)
    parser.add_argument("--w-target", type=float, default=1.0)
    parser.add_argument("--w-asap", type=float, default=1.0)
    parser.add_argument("--save-mesh", action="store_true", help="評価用にノード位置を保存（val 用）")
    parser.add_argument("--workers", type=int, default=1,
                        help="並列プロセス数。1プロセスあたり分解に約4〜8GBのメモリを使う")
    parser.add_argument("--limit", type=int, default=None, help="試しに先頭の何枚だけ処理する")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)
    if args.save_mesh:
        os.makedirs(os.path.join(args.out_dir, "meshes"), exist_ok=True)

    with open(args.ann) as f:
        coco = json.load(f)
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    images = coco["images"][:args.limit] if args.limit else coco["images"]
    # 同じ大きさの画像を続けて処理すると、行列の分解の作り直しが減る
    images = sorted(images, key=lambda im: (im["height"], im["width"], im["id"]))
    jobs = [(im, anns_by_img[im["id"]]) for im in images]

    cfg = {"img_dir": args.img_dir, "out_dir": args.out_dir, "box": args.box,
           "thresh": args.thresh, "scale_expand": args.scale_expand,
           "scale_shrink": args.scale_shrink, "w_target": args.w_target,
           "w_asap": args.w_asap, "save_mesh": args.save_mesh}

    new_images, new_anns, logs = [], [], []
    t0 = time.perf_counter()
    with Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for n, (info, anns, log) in enumerate(pool.imap(process_one, jobs, chunksize=4), 1):
            new_images.append(info)
            new_anns.extend(anns)
            logs.append(log)
            if n % 100 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} 枚, 経過 {time.perf_counter() - t0:.0f}s", flush=True)

    # 元の json の構成を保ち、images と annotations だけ差し替える
    out = dict(coco)
    out["images"] = sorted(new_images, key=lambda im: im["id"])
    out["annotations"] = sorted(new_anns, key=lambda a: a["id"])
    with open(os.path.join(args.out_dir, "annotations.json"), "w") as f:
        json.dump(out, f)
    with open(os.path.join(args.out_dir, "settings.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    region = np.array([lg["region"] for lg in logs])
    print(f"完了: {len(new_images)} 枚, ボックス {len(new_anns)} 個")
    print(f"  拡大領域の割合: 平均 {region.mean() * 100:.1f}%（最小 {region.min() * 100:.1f}%, "
          f"最大 {region.max() * 100:.1f}%）")
    print(f"  裏返りのあった画像: {sum(lg['n_fold'] > 0 for lg in logs)} 枚")
    print(f"  変形後に幅か高さが1画素未満のボックス: {sum(lg['n_small'] for lg in logs)} 個"
          f"（mmdet の学習では読み込み時に除外される）")


if __name__ == "__main__":
    main()