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

メッシュの可視化（デバッグ用）
  build_forward_map(..., name=名前, show_img=True) とすると、DEBUG_DIR（このファイルと
  同じ場所の debug_mesh/）に次を保存する（make_expand_region と同じ使い方）。
    {名前}_mesh.png : ノード種別・変形メッシュ・セルの面積比
    {名前}_fold.png : 裏返った三角形をメッシュ上で赤く塗ったもの
  変形後画像への重ね（{名前}_fold_overlay.png）は画像が要るので、画像側
  （poisson_asap_image.py）から save_fold_overlay(warped, fmap, 名前) で保存する。

  裏返り（fold）とは: メッシュの三角形の向きが変形前と逆になった所。
  つまり「上にあった点と下にあった点（左右でも同様）の位置関係が入れ替わった」状態で、
  三角形の符号付き面積が 0 以下になることで判定する（count_folds / triangle_areas）。

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

# メッシュの可視化（show_img=True のとき）の保存先: このファイルと同じ場所の debug_mesh/
DEBUG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_mesh")


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


def triangle_areas(mesh):
    """
    各四角形セルを2つの三角形（左上側・右下側）に分け、それぞれの符号付き面積を返す。
    mesh: (N, M, 2) ノードの位置 (x, y)
    Returns: (a1, a2) どちらも (N-1, M-1)。変形前はどちらも 0.5
    （count_folds と、可視化の面積比 cell_area_ratio で共通に使う）
    """
    p00, p01 = mesh[:-1, :-1], mesh[:-1, 1:]
    p10, p11 = mesh[1:, :-1], mesh[1:, 1:]

    def area(a, b, c):
        # 外積の半分 = 三角形 abc の符号付き面積（向きが変形前と同じなら正）
        return 0.5 * ((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
                      - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))

    a1 = area(p00, p01, p10)   # 左上側の三角形
    a2 = area(p11, p10, p01)   # 右下側の三角形
    return a1, a2


def count_folds(mesh):
    """
    各四角形を2つの三角形（左上側・右下側）に分け、符号付き面積が 0 以下のものを数える。
    0 以下 = 裏返り（または潰れ）。逆変換が一意に決まらなくなる。
    Returns: (裏返った三角形の数, 最小の符号付き面積)  ※変形前の面積は 0.5
    """
    a1, a2 = triangle_areas(mesh)
    return int((a1 <= 0).sum() + (a2 <= 0).sum()), float(min(a1.min(), a2.min()))


# ===========================================================================
# メッシュの可視化（デバッグ用。show_img=True のときに DEBUG_DIR へ保存）
# ===========================================================================
def plot_mesh(ax, X, Y, color, label, style='-', step=1):
    """ (N, M) の2次元座標配列から網目を描画するヘルパー（元コードと同じ。step 本おきに描く） """
    N, M = X.shape
    for r in range(0, N, step):
        ax.plot(X[r, :], Y[r, :], color=color, linestyle=style, alpha=0.6, lw=0.5)
    for c in range(0, M, step):
        ax.plot(X[:, c], Y[:, c], color=color, linestyle=style, alpha=0.6, lw=0.5)
    ax.plot([], [], color=color, linestyle=style, label=label)


def cell_area_ratio(mesh):
    """
    各四角形セルの「変形後の面積 / 変形前の面積」(N-1, M-1)。
    変形前のセルは 1x1 なので、2つの三角形の符号付き面積の和がそのまま面積比になる。
    1 より大きい = 拡大されたセル、1 より小さい = 縮小されたセル、0 以下 = 裏返り・潰れ
    """
    a1, a2 = triangle_areas(mesh)
    return a1 + a2


def save_mesh_debug(fmap, name="mesh", step=10, save_dir=DEBUG_DIR):
    """
    順写像のメッシュを可視化して {save_dir}/{name}_mesh.png に保存する。
      左  : ノード種別（0:固定, 1:拡大, 2:縮小）
      中央: 変形前の格子（灰色の破線）と変形後のメッシュ dst（赤）を step 本おきに重ねたもの
      右  : セルごとの面積比を log2 で表示（赤=拡大, 青=縮小, 黒=裏返り）
    fmap: build_forward_map の出力（ForwardMap）
    step: 網目を何本おきに描くか（全部描くと線が密すぎて見えないため）
    """
    import matplotlib
    matplotlib.use("Agg")   # 画面表示（plt.show）が使えない環境なので画像として保存する
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    out_h, out_w = fmap.content_size

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    # --- 左: ノード種別（3色に固定して、どの値がどの色かぶれないようにする）---
    ax = axes[0]
    cmap_types = matplotlib.colors.ListedColormap(["black", "tab:red", "tab:blue"])
    ax.imshow(fmap.node_types, cmap=cmap_types, vmin=-0.5, vmax=2.5, interpolation="nearest")
    ax.set_title("node types (black:fixed  red:expand  blue:shrink)")

    # --- 中央: 変形前の格子と変形後のメッシュを重ねる ---
    ax = axes[1]
    X0, Y0 = np.meshgrid(np.arange(out_w), np.arange(out_h))   # 変形前は dst = (c, r)
    plot_mesh(ax, X0, Y0, 'gray', 'Original grid', style='--', step=step)
    plot_mesh(ax, fmap.dst[..., 0], fmap.dst[..., 1], 'red', 'Deformed (dst)', step=step)
    ax.set_xlim(-0.5, out_w - 0.5)
    ax.set_ylim(out_h - 0.5, -0.5)   # y を下向きにする（画像と同じ向き）
    ax.set_aspect('equal')
    ax.set_title(f"Forward Mesh (every {step}th line)")
    ax.legend(loc="upper right")

    # --- 右: セルごとの面積比（log2 なので 0 = 変化なし, +1 = 2倍, -1 = 半分）---
    ax = axes[2]
    ratio = cell_area_ratio(fmap.dst)
    log_ratio = np.full(ratio.shape, np.nan)       # 裏返り（面積 0 以下）は log が取れないので NaN
    valid = ratio > 0
    log_ratio[valid] = np.log2(ratio[valid])
    cmap_ratio = matplotlib.colormaps["coolwarm"].copy()
    cmap_ratio.set_bad("black")                    # NaN（裏返り）を黒で表示
    lim = 2.0                                      # 表示範囲 1/4 倍〜4 倍
    im = ax.imshow(log_ratio, cmap=cmap_ratio, vmin=-lim, vmax=lim, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.035, label="log2(area ratio)")
    ax.set_title(f"cell area ratio (folds={fmap.n_fold}, min area={fmap.min_area:.3f})")

    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{name}_mesh.png")
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 裏返り（fold）の可視化
# ---------------------------------------------------------------------------
def folded_triangles(mesh):
    """
    変形後メッシュ mesh (N, M, 2) から、裏返った三角形の頂点だけを取り出す。
    各セルの左上側・右下側の2つの三角形について、符号付き面積が 0 以下（= 向きが反転）
    のものを集める。count_folds と同じ判定。
    Returns: (K, 3, 2) 裏返った三角形 K 個の頂点 (x, y)。無ければ (0, 3, 2)。
    """
    p00, p01 = mesh[:-1, :-1], mesh[:-1, 1:]
    p10, p11 = mesh[1:, :-1], mesh[1:, 1:]
    a1, a2 = triangle_areas(mesh)
    parts = []
    m1 = a1 <= 0                                      # 左上側の三角形が裏返ったセル
    if m1.any():
        parts.append(np.stack([p00[m1], p01[m1], p10[m1]], axis=1))   # (k1, 3, 2)
    m2 = a2 <= 0                                      # 右下側の三角形が裏返ったセル
    if m2.any():
        parts.append(np.stack([p11[m2], p10[m2], p01[m2]], axis=1))   # (k2, 3, 2)
    if parts:
        return np.concatenate(parts, axis=0)
    return np.empty((0, 3, 2), dtype=mesh.dtype)


def save_fold_mesh_debug(fmap, name="mesh", step=10, save_dir=DEBUG_DIR):
    """
    変形後メッシュ dst を薄く描いた上に、裏返った三角形を赤く塗って
    {save_dir}/{name}_fold.png に保存する。どこで裏返ったかがひと目でわかる。
    step: 背景に描くメッシュ線の間隔（塗りつぶしは裏返り全部を描く）
    """
    import matplotlib
    matplotlib.use("Agg")   # 画面表示が使えない環境なので画像として保存する
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    os.makedirs(save_dir, exist_ok=True)
    out_h, out_w = fmap.content_size

    fig, ax = plt.subplots(figsize=(14, 9))
    # 背景: 変形後メッシュを薄い灰色で（step 本おき。場所の目印用）
    plot_mesh(ax, fmap.dst[..., 0], fmap.dst[..., 1], 'gray', 'Deformed mesh', step=step)
    # 裏返った三角形を赤で塗る（多数あっても PolyCollection でまとめて描く）
    tris = folded_triangles(fmap.dst)
    if len(tris):
        pc = PolyCollection(tris, facecolors='red', edgecolors='darkred',
                            linewidths=0.3, alpha=0.7)
        ax.add_collection(pc)
        ax.plot([], [], color='red', label=f'Folded triangles ({len(tris)})')
    ax.set_xlim(-0.5, out_w - 0.5)
    ax.set_ylim(out_h - 0.5, -0.5)   # y を下向きに（画像と同じ向き）
    ax.set_aspect('equal')
    ax.set_title(f"folded triangles on deformed mesh "
                 f"(n_fold={fmap.n_fold}, min area={fmap.min_area:.3f})")
    ax.legend(loc='upper right')

    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{name}_fold.png")
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def fold_mask(fmap):
    """
    裏返った三角形の内部を 1、それ以外を 0 にした、内容サイズ（= 変形後画像と同じ大きさ）
    の 0/1 マスクを作る。dst は (x, y) の画素座標なので、そのまま cv2.fillPoly で塗れる。
    Returns: (out_h, out_w) の uint8 マスク
    """
    out_h, out_w = fmap.content_size
    mask = np.zeros((out_h, out_w), np.uint8)
    tris = folded_triangles(fmap.dst)
    if len(tris):
        polys = [np.round(t).astype(np.int32) for t in tris]   # 画素の整数座標にして塗る
        cv2.fillPoly(mask, polys, 1)
    return mask


def save_fold_overlay(warped, fmap, name="mesh", alpha=0.5, save_dir=DEBUG_DIR):
    """
    変形後画像 warped に、裏返りの起きた領域を半透明の赤で重ねて
    {save_dir}/{name}_fold_overlay.png に保存する。
    warped は nonuniform_resize の出力（内容サイズ）で、fmap と大きさが一致している前提。
    alpha: 赤の濃さ（0〜1）。
    """
    os.makedirs(save_dir, exist_ok=True)
    mask = fold_mask(fmap).astype(bool)
    # グレー画像なら色を重ねられるよう BGR に変換してから重ねる
    over = warped.copy() if warped.ndim == 3 else cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    red = np.zeros_like(over)
    red[..., 2] = 255   # BGR の R
    over[mask] = (alpha * red[mask] + (1 - alpha) * over[mask]).astype(np.uint8)
    out_path = os.path.join(save_dir, f"{name}_fold_overlay.png")
    cv2.imwrite(out_path, over)
    return out_path


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
                      scale_expand=2.0, scale_shrink=0.5, w_target=1.0, w_asap=1.0,
                      name=None, show_img=False):
    """
    region_map: get_expand_region(元画像) の出力。元画像と同じ (H, W) の 0/1 マップ（0:縮小, 1:拡大）
    box_size  : 縮小の枠 (幅, 高さ)。既定 800x800
    thresh    : マップ縮小時の閾値（面積平均した割合がこれ以上なら拡大）
    name      : 可視化画像のファイル名に使う名前（None なら "mesh"）
    show_img  : True のとき、メッシュの可視化（{name}_mesh.png）と裏返りの可視化
                （{name}_fold.png）を DEBUG_DIR に保存する
    使用例:
        region_map = get_expand_region(img)
        fmap = build_forward_map(region_map, box_size=(800, 800))
        fmap = build_forward_map(region_map, name="0000001", show_img=True)   # 可視化も保存
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
    fmap = ForwardMap(in_size=(in_h, in_w), content_size=(out_h, out_w), scale=scale,
                      node_types=node_types, src=src, dst=dst, n_fold=n_fold, min_area=min_area)
    # デバッグ用: メッシュと裏返りの可視化を DEBUG_DIR に保存する
    if show_img:
        vis_name = name if name is not None else "mesh"
        save_mesh_debug(fmap, vis_name)        # ノード種別・変形メッシュ・面積比
        save_fold_mesh_debug(fmap, vis_name)   # 裏返った三角形を塗ったメッシュ
    return fmap


# ===========================================================================
# 動作確認（元コードと同じく run_and_visualize と __main__）
# ===========================================================================
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

    # メッシュと裏返りの可視化（show_img=True と同じ処理）を最後の1枚で確認する -> debug_mesh/ に保存
    print(f"メッシュの可視化を保存しました: {save_mesh_debug(fmap, 'dummy')}")
    print(f"裏返りの可視化を保存しました: {save_fold_mesh_debug(fmap, 'dummy')}")

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