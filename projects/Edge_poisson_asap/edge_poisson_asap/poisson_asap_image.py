"""
画像の非一様縮小（全体の手順7）と、比較用の一様縮小

  - warp_image        : 逆写像 (map_x, map_y) を使い、元画像から直接読み取って
                        非一様縮小した画像を作る（cv2.remap）
  - resize_uniform    : 比較用の一様縮小（同じ枠・同じ補間。mmdet の Resize と同じ処理）
  - nonuniform_resize : 1枚の画像について、マップ -> 順写像 -> 逆写像 -> remap をまとめて行う
  - edge_nonuniform_resize : 画像だけを渡すと、make_expand_region.get_expand_region で
                        マップを作ってから nonuniform_resize を行う（実際に使う入口）

32 の倍数へのパディングはここでは行わない（検出器に入れる直前に mmdet 側で行う）。

動作確認（projects/Edge_poisson_asap/ で実行）:
    python -m edge_poisson_asap.poisson_asap_image --image 画像 [画像 ...]
  --image には画像ファイルかフォルダを1つ以上指定する（フォルダなら中の画像を名前順に --limit 枚）。
  結果の画像はこのファイルと同じ場所の deform_images/ に保存する（--out-dir で変更可）。
"""
import argparse
import os
import time

import cv2
import numpy as np

# 同じパッケージ内のファイルを相対 import する
from .make_expand_region import get_expand_region      # 利用者のマップ作成（元画像 -> 0/1）
from .poisson_asap_warp import build_forward_map, compute_content_size
from .poisson_asap_remap import build_inverse_map


# ===========================================================================
# 手順7: remap と一様縮小
# ===========================================================================
def warp_image(img, map_x, map_y, interpolation=cv2.INTER_LINEAR):
    """
    出力画素 (r, c) に、元画像の (map_x[r, c], map_y[r, c]) の位置の値を入れる。
    map_x, map_y は build_inverse_map の出力（元画像の座標, float32）。
    出力の大きさは map の大きさ（= 内容サイズ）になる。

    interpolation: 元画像の画素の間を読むときの補間。既定は双線形（一様縮小と同じ）。
    borderMode   : 参照先は外周固定のため元画像の内側に収まるが、画素の端ぎりぎりを
                   読むときに外側の値が混ざらないよう、端の画素を延長して使う。
    注意: 縮小領域では元画像を数画素おきに読むので、双線形補間では間の画素の情報が
          使われず、細かい模様がギザギザになる（エイリアシング）ことがある。
    """
    return cv2.remap(img, map_x, map_y, interpolation=interpolation,
                     borderMode=cv2.BORDER_REPLICATE)


def resize_uniform(img, box_size=(800, 800), interpolation=cv2.INTER_LINEAR):
    """
    比較用の一様縮小。非一様縮小と同じ枠・同じ丸め方で内容サイズを決め、cv2.resize で縮小する。
    （mmdet の Resize(keep_ratio=True) も cv2.resize の双線形補間を使う）
    """
    in_h, in_w = img.shape[:2]
    out_h, out_w, _ = compute_content_size(in_h, in_w, box_size)
    return cv2.resize(img, (out_w, out_h), interpolation=interpolation)


def nonuniform_resize(img, region_map, box_size=(800, 800), thresh=0.5,
                      scale_expand=2.0, scale_shrink=0.5, w_target=1.0, w_asap=1.0,
                      inverse_method="fast", interpolation=cv2.INTER_LINEAR):
    """
    1枚の画像を非一様縮小する（手順2〜7をまとめて実行）。
    img       : 元画像 (H, W) または (H, W, C)
    region_map: get_expand_region(img) の出力。img と同じ (H, W) の 0/1 マップ
    Returns:
      warped : 非一様縮小した画像（内容サイズ）
      fmap   : ForwardMap（順写像。正解ボックスの変換に使う）
      maps   : (map_x, map_y) 逆写像（検出結果を元画像の座標に戻すのに使う）
    """
    if region_map.shape[:2] != img.shape[:2]:
        raise ValueError(f"マップの大きさ {region_map.shape[:2]} が画像 {img.shape[:2]} と違います")
    # 手順2〜5: 順写像（各ノードの元画像上の位置 src と、変形後の位置 dst）
    fmap = build_forward_map(region_map, box_size, thresh, scale_expand, scale_shrink,
                             w_target, w_asap)
    # 手順6: 逆写像（出力の各画素が読む元画像の座標）
    map_x, map_y = build_inverse_map(fmap, method=inverse_method)
    # 手順7: 元画像から読み取って非一様縮小
    warped = warp_image(img, map_x, map_y, interpolation)
    return warped, fmap, (map_x, map_y)


def edge_nonuniform_resize(img, box_size=(800, 800), **kwargs):
    """
    画像だけを渡して非一様縮小する入口。
      1. get_expand_region(img) で元画像サイズの 0/1 マップを作る（make_expand_region.py）
      2. nonuniform_resize でマップ -> 順写像 -> 逆写像 -> remap を行う
    img   : cv2.imread で読んだ元画像（BGR, uint8）
    kwargs: nonuniform_resize の引数（thresh, scale_expand, scale_shrink など）
    Returns: warped, fmap, (map_x, map_y), region_map
    """
    region_map = get_expand_region(img)
    warped, fmap, maps = nonuniform_resize(img, region_map, box_size, **kwargs)
    return warped, fmap, maps, region_map


# ===========================================================================
# 動作確認
# ===========================================================================
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def collect_images(paths, limit):
    """ファイルはそのまま、フォルダは中の画像を名前順に limit 枚取り出す"""
    files = []
    for p in paths:
        if os.path.isdir(p):
            names = sorted(f for f in os.listdir(p) if f.lower().endswith(IMAGE_EXTS))
            files += [os.path.join(p, f) for f in names[:limit]]
        else:
            files.append(p)
    return files


def overlay_region(img, region_map, alpha=0.4):
    """マップの 1（拡大領域）を赤く重ねた画像（確認用）"""
    color = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    red = np.zeros_like(color)
    red[..., 2] = 255   # BGR の R
    mask = region_map.astype(bool)
    color[mask] = (alpha * red[mask] + (1 - alpha) * color[mask]).astype(np.uint8)
    return color


def save_comparison(path, img, region_map, uniform, warped):
    """元画像（拡大領域を赤で表示）、一様縮小、非一様縮小を横に並べて保存する"""
    import matplotlib
    matplotlib.use("Agg")   # 画面表示が使えない環境なのでファイルに保存する
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(24, 5))
    panels = [(overlay_region(img, region_map), "original + expand region (red)"),
              (uniform, f"uniform {uniform.shape[1]}x{uniform.shape[0]}"),
              (warped, f"poisson+ASAP {warped.shape[1]}x{warped.shape[0]}")]
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))   # matplotlib は RGB
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="非一様縮小（remap）の動作確認")
    parser.add_argument("--image", nargs="+", required=True,
                        help="元画像のファイルまたはフォルダ（複数可）")
    parser.add_argument("--limit", type=int, default=3, help="フォルダから取り出す枚数（既定 3）")
    parser.add_argument("--box", type=int, nargs=2, default=[800, 800], metavar=("W", "H"),
                        help="縮小の枠（幅 高さ）。既定 800 800")
    parser.add_argument("--out-dir",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "deform_images"),
                        help="結果の保存先。既定はこのファイルと同じ場所の deform_images/")
    args = parser.parse_args()
    box_size = tuple(args.box)
    os.makedirs(args.out_dir, exist_ok=True)

    files = collect_images(args.image, args.limit)
    if not files:
        raise FileNotFoundError(f"画像が見つかりません: {args.image}")

    for n, path in enumerate(files):
        img = cv2.imread(path)            # BGR, uint8
        if img is None:
            print(f"[スキップ] 画像を読めません: {path}")
            continue
        name = os.path.splitext(os.path.basename(path))[0]

        # --- マップ作成と非一様縮小（時間を分けて測る）---
        t0 = time.perf_counter()
        region_map = get_expand_region(img,name,show_img=True)
        t1 = time.perf_counter()
        warped, fmap, _ = nonuniform_resize(img, region_map, box_size)
        t2 = time.perf_counter()
        uniform = resize_uniform(img, box_size)
        print(f"[{name}] 元画像 {img.shape[1]}x{img.shape[0]} -> {warped.shape[1]}x{warped.shape[0]}, "
              f"拡大領域 {region_map.mean() * 100:.1f}%, 裏返り={fmap.n_fold}, "
              f"マップ作成 {t1 - t0:.2f}s, 非一様縮小 {t2 - t1:.2f}s（サイズの初回は分解を含む）")

        # --- 1枚目だけ: 変形しない設定の remap が一様縮小（cv2.resize）と一致するか ---
        if n == 0:
            same, _, _ = nonuniform_resize(img, region_map, box_size,
                                           scale_expand=1.0, scale_shrink=1.0)
            diff = np.abs(same.astype(int) - uniform.astype(int))
            print(f"    無変形の remap と一様縮小の差: 最大 {diff.max()}, 平均 {diff.mean():.3f}（画素値 0〜255）")

        # --- 保存: 個別の画像（拡大して見る用）と並べた図 ---
        cv2.imwrite(os.path.join(args.out_dir, f"{name}_uniform.png"), uniform)
        cv2.imwrite(os.path.join(args.out_dir, f"{name}_warped.png"), warped)
        save_comparison(os.path.join(args.out_dir, f"{name}_remap_check.png"),
                        img, region_map, uniform, warped)
    print(f"保存先: {args.out_dir}（*_remap_check.png, *_uniform.png, *_warped.png）")


if __name__ == "__main__":
    main()