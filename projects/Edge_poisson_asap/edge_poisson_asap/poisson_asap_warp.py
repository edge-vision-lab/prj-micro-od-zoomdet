"""
ポアソン変形 + ASAP による非一様縮小の前処理（順写像と逆写像の作成まで）

全体の流れ
  1. get_expand_region(元画像) で、元画像サイズの 0/1 マップを得る（make_expand_region.py）
       0 = 縮小領域, 1 = 拡大領域
  2. 縮小後のサイズ（内容サイズ）を決める                       … このファイル
       枠 box_size（既定 800x800）に収まる最大サイズまで縦横比を保って縮小
  3. マップを内容サイズへ縮小する（面積平均 -> 閾値で 0/1）       … このファイル
  4. 元コードのノード種別（0:固定, 1:拡大, 2:縮小）に変換する     … このファイル
       外周は固定(0)、マップ1 -> 拡大(1)、マップ0 -> 縮小(2)
  5. 行列計算で各ノードの変形後の位置を求める（順写像）          … poisson_asap_solver.py
  6. 逆写像（出力の各画素が参照する元画像の座標）を作る          … poisson_asap_remap.py
       -> cv2.remap(元画像, map_x, map_y, ...) で非一様縮小した画像が得られる

座標の約束
  - 座標はすべて (x, y) の順。x は横（列）、y は縦（行）、y は下向き。
  - 画素の中心を整数座標とする（cv2.resize / cv2.remap と同じ約束）。
  - メッシュのノード (r, c) は内容サイズの画素 (r, c) に 1対1 で対応する。
  - dst（変形後の位置）は内容サイズの画素座標。変形しなければ dst = (c, r)。
  - src（元画像上の位置）は一様縮小と同じ対応で決まり、変形によって変わらない。

置き場所: projects/Edge_poisson_asap/edge_poisson_asap/（パッケージ）
動作確認: projects/Edge_poisson_asap/ で
            python -m edge_poisson_asap.poisson_asap_warp
          を実行（図はこのファイルと同じ場所に保存）。
          パッケージ内で相対 import しているので、python edge_poisson_asap/poisson_asap_warp.py
          のようにファイルを直接実行すると import エラーになる。
他のコード（tools/ など）からの使い方:
    from edge_poisson_asap.poisson_asap_warp import build_forward_map, build_inverse_map
"""
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

# 同じパッケージ（edge_poisson_asap/）内のファイルを相対 import する
from .poisson_asap_solver import EXPAND, SHRINK, FIXED, border_mask, deform_mesh
from .poisson_asap_remap import build_inverse_map


# ===========================================================================
# 前処理（手順2〜4）と補助関数
# ===========================================================================
def compute_content_size(in_h, in_w, box_size=(800, 800)):
    """
    枠 box_size = (幅, 高さ) に収まる最大サイズまで、縦横比を保って拡大縮小したサイズ。
    丸め方は mmcv の rescale_size と同じ int(x * scale + 0.5)。
    （元画像が枠より小さい場合は拡大になる。VisDrone では 480x360 の1枚が該当）
    Returns: (out_h, out_w, scale)
    """
    box_w, box_h = box_size
    scale = min(box_w / in_w, box_h / in_h)
    return int(in_h * scale + 0.5), int(in_w * scale + 0.5), scale


def resize_region_map(region_map, out_h, out_w, thresh=0.5):
    """
    元画像サイズの 0/1 マップを内容サイズへ縮小する。
    面積平均（cv2.INTER_AREA）で「各出力画素に含まれる 1 の割合」を求め、
    thresh 以上を 1、未満を 0 にする。
    注意: uint8 のまま INTER_AREA をかけると割合が 0/1 に丸められるので、float32 にしてから縮小する。
    """
    values = np.unique(region_map)
    if not np.isin(values, [0, 1]).all():
        raise ValueError(f"マップの値は 0/1 である必要があります（実際の値: {values}）")
    ratio = cv2.resize(region_map.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_AREA)
    return (ratio >= thresh).astype(np.uint8)


def to_node_types(binary_map):
    """
    0/1 マップ（0:縮小, 1:拡大）を元コードのノード種別に変換する。
      マップ 1 -> EXPAND(1), マップ 0 -> SHRINK(2), 外周 -> FIXED(0)
    外周はマップの値にかかわらず固定にする。
    """
    types = np.where(binary_map == 1, EXPAND, SHRINK).astype(np.int8)
    types[border_mask(*types.shape)] = FIXED
    return types


def node_src_coords(in_h, in_w, out_h, out_w):
    """
    各ノード（= 内容サイズの画素）が対応する元画像上の位置 (N, M, 2)。
    一様縮小（cv2.resize）と同じ「画素中心どうしを合わせる」対応:
      x_src = (c + 0.5) * in_w / out_w - 0.5
    """
    xs = (np.arange(out_w) + 0.5) * in_w / out_w - 0.5
    ys = (np.arange(out_h) + 0.5) * in_h / out_h - 0.5
    X, Y = np.meshgrid(xs, ys)
    return np.stack([X, Y], axis=-1)


def count_folds(mesh):
    """
    各四角形を2つの三角形（左上側・右下側）に分け、符号付き面積が 0 以下のものを数える。
    0 以下 = 裏返り（または潰れ）。逆変換が一意に決まらなくなる。
    Returns: (裏返った三角形の数, 最小の符号付き面積)  ※変形前の面積は 0.5
    """
    p00, p01 = mesh[:-1, :-1], mesh[:-1, 1:]
    p10, p11 = mesh[1:, :-1], mesh[1:, 1:]

    def area(a, b, c):
        return 0.5 * ((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
                      - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))

    a1 = area(p00, p01, p10)
    a2 = area(p11, p10, p01)
    return int((a1 <= 0).sum() + (a2 <= 0).sum()), float(min(a1.min(), a2.min()))


# ===========================================================================
# まとめ: 1枚分の順写像を作る
# ===========================================================================
@dataclass
class ForwardMap:
    in_size: tuple          # 元画像の (高さ, 幅)
    content_size: tuple     # 内容サイズ（メッシュ）の (高さ, 幅)
    scale: float            # 一様縮小の倍率（内容サイズ / 元画像）
    node_types: np.ndarray  # (N, M) ノード種別 0:固定, 1:拡大, 2:縮小
    src: np.ndarray         # (N, M, 2) 各ノードの元画像上の位置 (x, y)
    dst: np.ndarray         # (N, M, 2) 各ノードの変形後の位置 (x, y)（内容サイズの画素座標）
    n_fold: int             # 裏返った三角形の数（0 が正常）
    min_area: float         # 最小の符号付き面積


def build_forward_map(region_map, box_size=(800, 800), thresh=0.5,
                      scale_expand=2.0, scale_shrink=0.5, w_target=1.0, w_asap=1.0):
    """
    region_map: get_expand_region(元画像) の出力。元画像と同じ (H, W) の 0/1 マップ（0:縮小, 1:拡大）
    box_size  : 縮小の枠 (幅, 高さ)。既定 800x800
    thresh    : マップ縮小時の閾値（面積平均した割合がこれ以上なら拡大）
    使用例:
        region_map = get_expand_region(img)
        fmap = build_forward_map(region_map, box_size=(800, 800))
    """
    in_h, in_w = region_map.shape[:2]
    # 手順2: 枠に収まる縮小後のサイズ（= メッシュのサイズ）を決める
    out_h, out_w, scale = compute_content_size(in_h, in_w, box_size)
    # 手順3: 元画像サイズのマップを、面積平均＋閾値で縮小後のサイズの 0/1 マップにする
    small_map = resize_region_map(region_map, out_h, out_w, thresh)
    # 手順4: 0/1 マップを元コードのノード種別（0:固定, 1:拡大, 2:縮小）に直す
    node_types = to_node_types(small_map)
    # 手順5: ポアソン＋ASAP の行列計算で、各ノードの変形後の位置 dst を求める
    _, dst = deform_mesh((out_h, out_w), node_types, scale_expand, scale_shrink,
                         w_target, w_asap)
    # 各ノードの元画像上の位置 src（一様縮小と同じ対応。変形とは無関係に決まる）
    src = node_src_coords(in_h, in_w, out_h, out_w)
    # 裏返りの確認
    n_fold, min_area = count_folds(dst)
    return ForwardMap(in_size=(in_h, in_w), content_size=(out_h, out_w), scale=scale,
                      node_types=node_types, src=src, dst=dst, n_fold=n_fold, min_area=min_area)


# ===========================================================================
# 動作確認（元コードと同じく run_and_visualize と __main__）
# ===========================================================================
def plot_mesh(ax, X, Y, color, label, style='-', step=1):
    """ (N, M) の2次元座標配列から網目を描画するヘルパー（元コードと同じ。step 本おきに描く） """
    N, M = X.shape
    for r in range(0, N, step):
        ax.plot(X[r, :], Y[r, :], color=color, linestyle=style, alpha=0.6, lw=0.5)
    for c in range(0, M, step):
        ax.plot(X[:, c], Y[:, c], color=color, linestyle=style, alpha=0.6, lw=0.5)
    ax.plot([], [], color=color, linestyle=style, label=label)


def make_dummy_region(H, W, n_rect=40, size=50, seed=0):
    """get_expand_region の代わり: 元画像サイズのマップに正方形の拡大領域をランダムに置く"""
    rng = np.random.default_rng(seed)
    region = np.zeros((H, W), np.uint8)
    for _ in range(n_rect):
        y, x = rng.integers(0, H - size), rng.integers(0, W - size)
        region[y:y + size, x:x + size] = 1
    return region


def run_and_visualize():
    # 画面表示（plt.show）が使えない環境なので、描画用のバックエンドを指定して
    # このファイルと同じ場所に画像として保存する
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # VisDrone の代表的な 4:3 と 16:9 のサイズで確認（各2枚目は分解の使い回しで速くなるはず）
    cases = [((1050, 1400), 0), ((1050, 1400), 1), ((765, 1360), 2), ((765, 1360), 3)]
    fmap = None
    for (H, W), seed in cases:
        region = make_dummy_region(H, W, seed=seed)
        t0 = time.perf_counter()
        fmap = build_forward_map(region)
        dt = time.perf_counter() - t0
        print(f"{W}x{H} -> 内容 {fmap.content_size[1]}x{fmap.content_size[0]}: {dt:.2f}s, "
              f"裏返り={fmap.n_fold}, 最小面積={fmap.min_area:.3f}")

    # 逆写像（既定は高速版 method="fast"）を最後の1枚（16:9）で確認する
    t0 = time.perf_counter()
    map_x, map_y = build_inverse_map(fmap)
    print(f"逆写像: {time.perf_counter() - t0:.2f}s, "
          f"参照先の範囲 x={map_x.min():.1f}〜{map_x.max():.1f}, y={map_y.min():.1f}〜{map_y.max():.1f} "
          f"（元画像 {fmap.in_size[1]}x{fmap.in_size[0]}）")

    # 最後の1枚を描画
    fig, axes = plt.subplots(1, 3, figsize=(24, 5))
    axes[0].imshow(fmap.node_types)
    axes[0].set_title("node types (0:fixed 1:expand 2:shrink)")
    ax = axes[1]
    ax.set_title("Forward Mesh: dst (every 10th line)")
    plot_mesh(ax, fmap.dst[..., 0], fmap.dst[..., 1], 'red', 'Deformed', step=10)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.legend()
    # 逆写像: 出力の等間隔な画素が、元画像のどこを参照しているか（元コードの Inverse Remap Mesh）
    ax = axes[2]
    ax.set_title("Inverse Remap Mesh: source coords on original image (every 10th line)")
    plot_mesh(ax, map_x, map_y, 'red', 'Source Coords (map_x, map_y)', step=10)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forward_mesh_check.png")
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"図を保存しました: {out_path}")


if __name__ == "__main__":
    run_and_visualize()