"""
逆写像の作成（全体の手順6）

変形後のメッシュから「出力（縮小後）の各画素が、元画像のどこを参照するか」を求める。
  - generate_remap_grids      : 元コードと同じ名前・引数・戻り値・方法（3次補間）。遅い
  - generate_remap_grids_fast : 高速版。メッシュ自身の三角形の中で線形補間する
  - build_inverse_map         : 元画像の座標に直し、cv2.remap にそのまま渡せる形で返す
                                （method="fast" が既定、"cubic" で元コードの方法）

2つの方法の違い
  元コードの方法（cubic）:
    変形後のノードを「ばらばらの点の集まり」とみなし、三角形分割を一から作り直してから
    3次補間する。メッシュの並びが規則的であることを使っていないので遅い。
  高速版（fast）:
    変形後のメッシュの四角形を、count_folds と同じ分け方で2つの三角形に分け、
    各三角形が覆う出力画素を直接探して、三角形の中で線形補間（重心座標）する。
    三角形分割を作り直さないので速い。
    順写像（ノード間を直線でつないだ変形）をちょうど逆にたどる補間になる。
  両者の参照先の差（実測、元画像の画素単位）: 最大 0.2〜0.35、平均 0.002 程度
  （cubic は3次曲面、fast は三角形ごとの平面で補間するため）。
  時間（実測、作業環境）: 800x450 で cubic 5.2秒 -> fast 0.35秒、800x600 で 7.3秒 -> 0.52秒
"""
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator


def generate_remap_grids(orig_mesh, new_mesh):
    """
    出力空間における(N, M)の等間隔グリッドを作成し，
    対応する入力空間上の歪んだ座標(map_x, map_y)を逆補間して返す．（元コードと同じ）

    orig_mesh: (N, M, 2) 変形前のノード座標（内容サイズの画素座標）
    new_mesh : (N, M, 2) 変形後のノード座標（同上）
    Returns: grid_x, grid_y, map_x, map_y すべて (N, M)
      map_x, map_y は「内容サイズの画素座標」で表した参照先。元画像の座標に直すのは
      build_inverse_map で行う。

    補足:
      - 外周を固定しているので、変形後の座標の最小・最大は 0〜M-1, 0〜N-1 のまま。
        そのため linspace で作るグリッドはちょうど整数（= 出力の各画素の中心）になる。
      - 元コードの griddata(method='cubic') は内部で CloughTocher2DInterpolator を使う。
        元コードは x と y で2回呼んで三角形分割を2回作っていたので、
        ここでは x と y をまとめて1回で補間する（結果は同じ）。
      - 変形後のノードを三角形分割し、その中で3次補間する。外周の外は NaN になるが、
        外周固定なのでグリッドは必ず内側に入る。
    """
    N, M, _ = orig_mesh.shape
    new_coords = new_mesh.reshape(-1, 2)
    orig_coords = orig_mesh.reshape(-1, 2)

    min_x, max_x = new_coords[:, 0].min(), new_coords[:, 0].max()
    min_y, max_y = new_coords[:, 1].min(), new_coords[:, 1].max()

    # 出力空間上に元のメッシュサイズと同じ (N, M) の等間隔グリッドを生成
    grid_x, grid_y = np.meshgrid(np.linspace(min_x, max_x, M),
                                 np.linspace(min_y, max_y, N))

    # Deformed -> Original の逆写像を補間（x と y を一度に）
    interp = CloughTocher2DInterpolator(new_coords, orig_coords)
    mapped = interp(grid_x, grid_y)          # (N, M, 2)
    map_x, map_y = mapped[..., 0], mapped[..., 1]

    return grid_x, grid_y, map_x, map_y


def _split_into_triangles(mesh):
    """
    (N, M, 2) のメッシュの各四角形を2つの三角形に分け、頂点座標を (T, 3, 2) で返す。
    分け方は count_folds と同じ:
      三角形1 = (左上, 右上, 左下),  三角形2 = (右下, 左下, 右上)
    T = 2 * (N-1) * (M-1)
    """
    p00, p01 = mesh[:-1, :-1], mesh[:-1, 1:]   # 左上, 右上
    p10, p11 = mesh[1:, :-1], mesh[1:, 1:]     # 左下, 右下
    tri1 = np.stack([p00, p01, p10], axis=-2).reshape(-1, 3, 2)
    tri2 = np.stack([p11, p10, p01], axis=-2).reshape(-1, 3, 2)
    return np.concatenate([tri1, tri2], axis=0)


def generate_remap_grids_fast(orig_mesh, new_mesh, eps=1e-9):
    """
    generate_remap_grids の高速版（戻り値の形は同じ）。
    変形後のメッシュの各三角形について、その中に入る出力画素（整数座標）を探し、
    重心座標で変形前の座標を線形補間する。

    手順:
      1. 変形後のメッシュと変形前のメッシュを、同じ分け方で三角形に分ける
         （変形後の三角形 k と変形前の三角形 k は同じ3ノードでできている）
      2. 各変形後三角形を囲む長方形（外接矩形）に入る整数画素を候補にする
         三角形は小さい（1〜数画素）ので、候補は1三角形あたり数個〜十数個
      3. 候補画素ごとに重心座標 (l0, l1, l2) を求め、3つとも 0 以上なら三角形の中
         -> その画素の参照先 = l0*変形前の頂点0 + l1*頂点1 + l2*頂点2
      全三角形を同時に（配列で）処理し、for 文は外接矩形内の位置（dx, dy）についてだけ回す。

    eps: 三角形の辺の上にちょうど乗る画素（外周など）を取りこぼさないための許容誤差
    """
    N, M, _ = orig_mesh.shape
    dst_tri = _split_into_triangles(new_mesh)    # (T, 3, 2) 変形後（出力空間）
    src_tri = _split_into_triangles(orig_mesh)   # (T, 3, 2) 変形前（参照先の値）

    A, B, C = dst_tri[:, 0], dst_tri[:, 1], dst_tri[:, 2]   # 各 (T, 2)
    AB, AC = B - A, C - A
    # 符号付き面積の2倍。裏返った（<=0）三角形は逆写像が決まらないので使わない
    denom = AB[:, 0] * AC[:, 1] - AB[:, 1] * AC[:, 0]
    ok = denom > 0

    # 2. 外接矩形に入る整数画素の範囲 [x0, x1] x [y0, y1]（画像の外ははみ出さないよう切る）
    tri_min = dst_tri.min(axis=1)
    tri_max = dst_tri.max(axis=1)
    x0 = np.clip(np.ceil(tri_min[:, 0] - eps), 0, M - 1).astype(np.int64)
    y0 = np.clip(np.ceil(tri_min[:, 1] - eps), 0, N - 1).astype(np.int64)
    x1 = np.clip(np.floor(tri_max[:, 0] + eps), 0, M - 1).astype(np.int64)
    y1 = np.clip(np.floor(tri_max[:, 1] + eps), 0, N - 1).astype(np.int64)
    width = x1 - x0 + 1     # 0 以下なら、その三角形に入る整数画素はない
    height = y1 - y0 + 1

    map_x = np.full((N, M), np.nan)
    map_y = np.full((N, M), np.nan)

    # 外接矩形の中の位置 (dx, dy) ごとに、全三角形をまとめて処理する
    for dy in range(max(int(height.max()), 0)):
        for dx in range(max(int(width.max()), 0)):
            # この (dx, dy) が外接矩形の中にある三角形だけを対象にする
            cand = ok & (dx < width) & (dy < height)
            if not cand.any():
                continue
            k = np.where(cand)[0]
            px = (x0[k] + dx).astype(float)      # 候補画素の座標
            py = (y0[k] + dy).astype(float)

            # 3. 重心座標: P = A + l1*AB + l2*AC を解く（2次元の外積で計算）
            APx, APy = px - A[k, 0], py - A[k, 1]
            l1 = (APx * AC[k, 1] - APy * AC[k, 0]) / denom[k]
            l2 = (AB[k, 0] * APy - AB[k, 1] * APx) / denom[k]
            l0 = 1.0 - l1 - l2
            inside = (l0 >= -eps) & (l1 >= -eps) & (l2 >= -eps)
            if not inside.any():
                continue

            k, l0, l1, l2 = k[inside], l0[inside], l1[inside], l2[inside]
            # 同じ3つの割合で、変形前の頂点座標を混ぜる = この画素の参照先
            s = (l0[:, None] * src_tri[k, 0] + l1[:, None] * src_tri[k, 1]
                 + l2[:, None] * src_tri[k, 2])
            # 2つの三角形の境目にある画素は2回書かれるが、線形補間なので値は同じ
            r = (y0[k] + dy)
            c = (x0[k] + dx)
            map_x[r, c] = s[:, 0]
            map_y[r, c] = s[:, 1]

    # 戻り値の形を generate_remap_grids とそろえる（外周固定なので出力グリッドは整数格子）
    grid_x, grid_y = np.meshgrid(np.arange(M, dtype=float), np.arange(N, dtype=float))
    return grid_x, grid_y, map_x, map_y


def build_inverse_map(fmap, method="fast"):
    """
    ForwardMap から、cv2.remap にそのまま渡せる逆写像を作る。
    method: "fast"（既定。メッシュの三角形で線形補間）または "cubic"（元コードの方法）
    Returns: map_x, map_y  どちらも (内容高さ, 内容幅) の float32。値は元画像上の座標。
      出力画素 (r, c) の値 = 元画像の (map_x[r, c], map_y[r, c]) の位置の値
    使用例:
        map_x, map_y = build_inverse_map(fmap)
        out = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    """
    in_h, in_w = fmap.in_size
    out_h, out_w = fmap.content_size
    # 変形前のノード座標 = 内容サイズの整数格子
    X, Y = np.meshgrid(np.arange(out_w, dtype=float), np.arange(out_h, dtype=float))
    orig_mesh = np.stack([X, Y], axis=-1)

    # 内容サイズの画素座標で逆写像を求める
    if method == "fast":
        _, _, map_x, map_y = generate_remap_grids_fast(orig_mesh, fmap.dst)
    elif method == "cubic":
        _, _, map_x, map_y = generate_remap_grids(orig_mesh, fmap.dst)
    else:
        raise ValueError(f'method は "fast" か "cubic" です（指定: {method}）')
    if np.isnan(map_x).any() or np.isnan(map_y).any():
        raise RuntimeError("逆写像に NaN があります（どの三角形にも入らない画素があります。"
                           "メッシュの裏返りを確認してください）")

    # 内容サイズの座標 -> 元画像の座標（node_src_coords と同じ「画素中心を合わせる」対応）
    map_x = (map_x + 0.5) * in_w / out_w - 0.5
    map_y = (map_y + 0.5) * in_h / out_h - 0.5
    return map_x.astype(np.float32), map_y.astype(np.float32)