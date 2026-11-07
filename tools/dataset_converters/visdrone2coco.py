"""VisDrone2019-DET のアノテーションを COCO 形式の json に変換する。

ZoomDet 環境構築時に自作した同名スクリプト（プロジェクトの
zoomdet-setup-record.md の B-2）と同じ仕様で書き直したもの。原本の
ソースは手元に無いため、記録にある仕様と件数で照合している。

仕様
    - category 0 (ignored regions) と 11 (others) を除外し、10 クラスにする。
      --ignore-mode crowd を指定すると、category 0 を iscrowd=1 として残す
      （COCO 評価で、無視領域内の検出を誤検出として数えなくなる）。
      既定の drop は ZoomDet 実験と同じ扱い。
    - 空行と行末のカンマを許す。
    - 画像外にはみ出したボックスは画像内に切り詰め、幅か高さが 0 以下に
      なったものは捨てる。
    - アノテーションファイルが無い画像は、物体 0 個の画像として残す。

VisDrone の1行: x, y, w, h, score, category, truncation, occlusion

使い方
    python tools/dataset_converters/visdrone2coco.py \
        data/VisDrone/VisDrone2019-DET-val \
        data/VisDrone/VisDrone2019-DET-val/my_val.json
"""
import argparse
import json
import os
import os.path as osp

import cv2

CLASSES = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
           'tricycle', 'awning-tricycle', 'bus', 'motor')
# VisDrone の category 1..10 -> COCO の category_id 1..10（名前は CLASSES 順）
IMG_EXTS = ('.jpg', '.jpeg', '.png')


def parse_line(line):
    vals = [v for v in line.strip().split(',') if v.strip() != '']
    if len(vals) < 6:
        return None
    return [int(float(v)) for v in vals[:8]]


def convert(split_dir, out_file, ignore_mode='drop'):
    img_dir = osp.join(split_dir, 'images')
    ann_dir = osp.join(split_dir, 'annotations')
    names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS))

    images, annotations = [], []
    stats = dict(skipped_cat0=0, skipped_cat11=0, skipped_degenerate=0,
                 clipped=0, missing_ann=0, crowd=0)
    ann_id = 1
    for img_id, name in enumerate(names, start=1):
        h, w = cv2.imread(osp.join(img_dir, name)).shape[:2]
        images.append(dict(id=img_id, file_name=name, height=h, width=w))
        ann_path = osp.join(ann_dir, osp.splitext(name)[0] + '.txt')
        if not osp.exists(ann_path):
            stats['missing_ann'] += 1
            continue
        with open(ann_path) as f:
            for line in f:
                vals = parse_line(line)
                if vals is None:
                    continue
                x, y, bw, bh, _score, cat = vals[:6]
                crowd = 0
                if cat == 0:
                    if ignore_mode == 'crowd':
                        crowd = 1
                    else:
                        stats['skipped_cat0'] += 1
                        continue
                elif cat == 11:
                    stats['skipped_cat11'] += 1
                    continue
                elif not 1 <= cat <= 10:
                    stats['skipped_degenerate'] += 1
                    continue
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w, x + bw), min(h, y + bh)
                if (x1, y1, x2, y2) != (x, y, x + bw, y + bh):
                    stats['clipped'] += 1
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    stats['skipped_degenerate'] += 1
                    continue
                cw, ch = x2 - x1, y2 - y1
                annotations.append(dict(
                    id=ann_id, image_id=img_id,
                    # crowd 領域は category_id を 1 にしておく（評価では無視される）
                    category_id=cat if crowd == 0 else 1,
                    bbox=[x1, y1, cw, ch], area=cw * ch,
                    iscrowd=crowd, ignore=crowd,
                    truncation=vals[6] if len(vals) > 6 else 0,
                    occlusion=vals[7] if len(vals) > 7 else 0))
                stats['crowd'] += crowd
                ann_id += 1

    categories = [dict(id=i + 1, name=n) for i, n in enumerate(CLASSES)]
    os.makedirs(osp.dirname(osp.abspath(out_file)), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(dict(images=images, annotations=annotations,
                       categories=categories), f)

    normal = [a for a in annotations if a['iscrowd'] == 0]
    small = sum(a['area'] < 32 ** 2 for a in normal)
    skipped = stats['skipped_cat0'] + stats['skipped_cat11'] + stats['skipped_degenerate']
    print(f'images:{len(images)} annotations:{len(normal)} skipped:{skipped} '
          f'small_ratio:{small / max(len(normal), 1):.3f}')
    print('detail:', stats)
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('split_dir', help='images/ と annotations/ を含むディレクトリ')
    p.add_argument('out_file')
    p.add_argument('--ignore-mode', choices=['drop', 'crowd'], default='drop')
    a = p.parse_args()
    convert(a.split_dir, a.out_file, a.ignore_mode)


if __name__ == '__main__':
    main()
