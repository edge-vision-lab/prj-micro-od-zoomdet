"""
ポアソン変形 + ASAP の行列計算（全体の手順5）

元コード deform_mesh を部品に分け、画像に依存しない部分（行列 A とその LU 分解）を
サイズごとに一度だけ作って使い回すようにしたもの。

元コードの for 文との対応（for 文を numpy の配列計算に置き換えた。作られる行列は同じ）
  - 元コード 2. エッジリストの for 文       -> build_nodes_and_edges
  - 元コード 2. ターゲットエッジの for 文   -> MeshSolver.edge_scales（と solve 内の掛け算）
  - 元コード 3. M_mat に値を入れる for 文   -> build_target_matrix
  - 元コード 4. C_mat に値を入れる for 文   -> build_asap_matrix
  for 文をやめた理由: 800x600 では約48万ノード・約96万辺あり、
  1要素ずつ代入する for 文（lil_matrix）では組み立てだけで非常に時間がかかるため。
  - deform_mesh : 元コードと同じ名前・引数・戻り値
  - MeshSolver  : 行列の組み立てと分解を保持し、画像ごとには右辺だけ変えて解く

座標の約束: (x, y) の順、y は下向き、単位は縮小後（内容サイズ）の画素。
"""
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ノード種別（元コードと同じ番号を使う）
FIXED = 0    # 固定（外周）
EXPAND = 1   # 拡大
SHRINK = 2   # 縮小


# ===========================================================================
# 行列の組み立て（元コード deform_mesh の 1〜4, 6 を部品に分けたもの）
# ===========================================================================
def build_nodes_and_edges(N, M):
    """
    元コードの「1. 初期ノード座標の生成」と「2. エッジリストの作成」。

    元コードの for 文:
        for r in range(N):
            for c in range(M):
                i = r * M + c
                if c < M - 1: edges.append((i, i + 1))      # 水平エッジ（右隣）
                if r < N - 1: edges.append((i, i + M))      # 垂直エッジ（下隣）
    ここでは、ノード番号を (N, M) の表 idx に並べ、ずらして切り出すことで
    「全ノードとその右隣」「全ノードとその下隣」の組を一度に作る。
      例: N=2, M=3 のとき idx = [[0, 1, 2],
                                  [3, 4, 5]]
          idx[:, :-1] = [[0,1],[3,4]] と idx[:, 1:] = [[1,2],[4,5]]
            -> 水平エッジ (0,1) (1,2) (3,4) (4,5)
          idx[:-1, :] = [[0,1,2]]     と idx[1:, :] = [[3,4,5]]
            -> 垂直エッジ (0,3) (1,4) (2,5)

    Returns:
      orig_coords: (N*M, 2) 変形前のノード座標 (x, y)。単位は内容サイズの画素
      edge_i, edge_j: 各辺の始点・終点のノード番号（ノード番号 i = r * M + c）
      edge_vec: (E, 2) 変形前の辺ベクトル e = orig[j] - orig[i]
    辺の並びは元コードと異なる（水平をまとめてから垂直）が、
    行の順番が変わるだけなので正規方程式 A, b は同じになる。
    """
    # 1. 初期ノード座標（元コードと同じ）。ノード i = r*M + c の座標は (c, r)
    X, Y = np.meshgrid(np.arange(M, dtype=float), np.arange(N, dtype=float))
    orig_coords = np.column_stack([X.ravel(), Y.ravel()])

    # ノード番号を画像と同じ並び (N, M) にした表
    idx = np.arange(N * M).reshape(N, M)
    # 水平エッジ: 「右端の列を除いた全ノード」と「左端の列を除いた全ノード」を対にする
    h_i, h_j = idx[:, :-1].ravel(), idx[:, 1:].ravel()
    # 垂直エッジ: 「下端の行を除いた全ノード」と「上端の行を除いた全ノード」を対にする
    v_i, v_j = idx[:-1, :].ravel(), idx[1:, :].ravel()
    # 水平エッジをすべて並べた後に垂直エッジを並べる（edge_i[k], edge_j[k] が k 番目の辺）
    edge_i = np.concatenate([h_i, v_i])
    edge_j = np.concatenate([h_j, v_j])
    # 各辺の変形前のベクトル（元コードの e = orig_coords[j] - orig_coords[i] を全辺まとめて計算）
    edge_vec = orig_coords[edge_j] - orig_coords[edge_i]
    return orig_coords, edge_i, edge_j, edge_vec


def build_target_matrix(edge_i, edge_j, num_nodes):
    """
    元コードの「3. ターゲット制約行列 M_mat の構築」（ポアソン項）。
      行 2k   : x_j - x_i = target_x
      行 2k+1 : y_j - y_i = target_y
    （未知数ベクトルはノードごとに x, y が交互に並ぶ: [x_0, y_0, x_1, y_1, ...]
      なのでノード i の x は 2*i 列目、y は 2*i+1 列目）

    元コードの for 文:
        for k, (i, j) in enumerate(edges):
            M_mat[2k,   2j]   =  1;  M_mat[2k,   2i]   = -1
            M_mat[2k+1, 2j+1] =  1;  M_mat[2k+1, 2i+1] = -1
    1つの辺につき4つの値を入れている。ここでは「どの行・どの列に・何の値を入れるか」を
    全辺分まとめて3本の配列 rows, cols, vals に並べ、疎行列を一度に作る。
    sp.csr_matrix((vals, (rows, cols))) は「M[rows[n], cols[n]] = vals[n] を全 n について
    行った行列」を作るので、元コードの for 文と同じ行列になる。
    （右辺 t_vec は画像ごとに変わるので、ここではなく MeshSolver.solve で作る）
    """
    num_edges = edge_i.size
    k = np.arange(num_edges)          # 辺の番号 0, 1, ..., E-1
    ones = np.ones(num_edges)
    # 4つの塊は、元コードの4つの代入文にそれぞれ対応する
    #   [x行・jのx列に+1] [x行・iのx列に-1] [y行・jのy列に+1] [y行・iのy列に-1]
    rows = np.concatenate([2 * k,      2 * k,      2 * k + 1,      2 * k + 1])
    cols = np.concatenate([2 * edge_j, 2 * edge_i, 2 * edge_j + 1, 2 * edge_i + 1])
    vals = np.concatenate([ones,       -ones,      ones,           -ones])
    return sp.csr_matrix((vals, (rows, cols)), shape=(2 * num_edges, 2 * num_nodes))


def build_asap_matrix(N, M):
    """
    元コードの「4. ASAP制約行列 C_mat の構築」（コーシー・リーマン制約）。
    元コードと同じく、各四角形の左上の三角形（左上 i, 右上 R, 左下 D）のみに課す。
      1本目: -x_i + x_D - y_i + y_R = 0
      2本目:  x_i - x_R - y_i + y_D = 0

    元コードの for 文:
        for r in range(N - 1):
            for c in range(M - 1):              # 四角形 (r, c) ごとに
                i = r*M + c; R = i + 1; D = i + M
                1本目の4つの値を row 行目に入れ, row += 1
                2本目の4つの値を row 行目に入れ, row += 1
    ここでは build_target_matrix と同じく、全四角形分の (行, 列, 値) を配列に並べて一度に作る。
    四角形 q の1本目は 2q 行目、2本目は 2q+1 行目（元コードの row の進み方と同じ）。
    """
    idx = np.arange(N * M).reshape(N, M)
    # 四角形 (r, c) の3つの角のノード番号を、全四角形分まとめて取り出す
    #   左上 = idx[r, c], 右上 = idx[r, c+1], 左下 = idx[r+1, c]
    q_i = idx[:-1, :-1].ravel()   # 左上
    q_R = idx[:-1, 1:].ravel()    # 右上
    q_D = idx[1:, :-1].ravel()    # 左下
    nq = q_i.size                 # 四角形の数 (N-1)*(M-1)
    r = np.arange(nq)             # 四角形の番号
    one = np.ones(nq)

    # 1本目: 4つの塊は元コードの4つの代入（x_i に -1, x_D に +1, y_i に -1, y_R に +1）
    rows1 = np.concatenate([2 * r] * 4)
    cols1 = np.concatenate([2 * q_i, 2 * q_D, 2 * q_i + 1, 2 * q_R + 1])
    vals1 = np.concatenate([-one, one, -one, one])

    # 2本目: 4つの塊は元コードの4つの代入（x_i に +1, x_R に -1, y_i に -1, y_D に +1）
    rows2 = np.concatenate([2 * r + 1] * 4)
    cols2 = np.concatenate([2 * q_i, 2 * q_R, 2 * q_i + 1, 2 * q_D + 1])
    vals2 = np.concatenate([one, -one, -one, one])

    return sp.csr_matrix(
        (np.concatenate([vals1, vals2]),
         (np.concatenate([rows1, rows2]), np.concatenate([cols1, cols2]))),
        shape=(2 * nq, 2 * N * M))


def border_mask(N, M):
    """外周ノードを True にした (N, M) の配列。外周＝固定ノード。"""
    mask = np.zeros((N, M), dtype=bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
    return mask


def split_fixed_free(mask):
    """
    元コードの「6. 境界条件の適用」のうち、固定/自由ノードの番号と
    未知数ベクトル上の位置（x と y が交互に並ぶ）を求める部分。
    元コードも for 文ではなく np.where で同じことをしている（判定を node_types == 0 から
    外周かどうか mask に変えただけ）。
      idx_C, idx_U: ノード番号 n を未知数ベクトル上の位置 [2n, 2n+1] に展開したもの
    """
    fixed_nodes = np.where(mask.ravel())[0]
    free_nodes = np.where(~mask.ravel())[0]
    idx_C = np.ravel(np.column_stack([2 * fixed_nodes, 2 * fixed_nodes + 1]))
    idx_U = np.ravel(np.column_stack([2 * free_nodes, 2 * free_nodes + 1]))
    return fixed_nodes, free_nodes, idx_C, idx_U


# ===========================================================================
# 画像に依存しない部分をまとめて保持し、画像ごとには右辺だけ変えて解く
# ===========================================================================
class MeshSolver:
    """
    メッシュサイズ (N, M) と重みだけで決まる部分（行列 A とその分解）を一度だけ作る。
    固定ノードは常に外周なので A_UU は画像に依存せず、LU 分解を使い回せる。
    （元コードは spsolve で毎回分解していた）
    """

    def __init__(self, N, M, w_target=1.0, w_asap=1.0):
        if N < 3 or M < 3:
            raise ValueError("メッシュは縦横とも 3 ノード以上必要です（外周以外のノードがなくなるため）")
        self.N, self.M = N, M
        self.w_target, self.w_asap = w_target, w_asap

        # 1, 2: ノードと辺
        self.orig_coords, self.edge_i, self.edge_j, self.edge_vec = build_nodes_and_edges(N, M)
        # 3, 4: ポアソン項と ASAP 項の行列
        self.M_mat = build_target_matrix(self.edge_i, self.edge_j, N * M)
        self.C_mat = build_asap_matrix(N, M)
        # 5: 正規方程式の係数行列 A（画像に依存しない）
        A = self._build_normal_matrix()
        # 6: 境界条件（外周を固定）
        self.border = border_mask(N, M)
        self.fixed_nodes, self.free_nodes, self.idx_C, self.idx_U = split_fixed_free(self.border)
        self.v_C = np.ravel(self.orig_coords[self.fixed_nodes])   # 固定ノードは元の位置
        self.A_UC = A[self.idx_U, :][:, self.idx_C].tocsr()
        # 7 の前半: A_UU を一度だけ LU 分解
        self._factorize(A[self.idx_U, :][:, self.idx_U])

    def _build_normal_matrix(self):
        """元コードの「5. 正規方程式」の A。w_asap = 0 なら ASAP 項を入れない（元コードと同じ）。"""
        A = self.w_target * (self.M_mat.T @ self.M_mat)
        if self.w_asap > 0:
            A = A + self.w_asap * (self.C_mat.T @ self.C_mat)
        return A.tocsr()

    def _factorize(self, A_UU):
        """A_UU の LU 分解。800x600 で約1分・約4.3GB（実測）。"""
        t0 = time.perf_counter()
        self.lu = spla.splu(A_UU.tocsc(), permc_spec="MMD_AT_PLUS_A")
        self.factor_time = time.perf_counter() - t0

    def edge_scales(self, node_types, scale_expand, scale_shrink):
        """
        各辺の倍率（元コードの「2. ターゲットエッジの計算」と同じ規則）:
          両端が拡大 -> scale_expand, 両端が縮小 -> scale_shrink, それ以外 -> 1.0（無変形）
        固定ノード（外周）を含む辺は「それ以外」に当たるので 1.0 になる。

        元コードの for 文:
            for (i, j) in edges:
                scale = 1.0
                if type_i == 1 and type_j == 1: scale = scale_expand
                elif type_i == 2 and type_j == 2: scale = scale_shrink
                target_edges.append(e * scale)
        ここでは全辺の倍率を配列で一度に決める（e * scale の掛け算は solve で行う）。
        """
        t = node_types.ravel()
        # 全辺について、始点と終点のノード種別を取り出す（元コードの type_i, type_j）
        ti, tj = t[self.edge_i], t[self.edge_j]
        # まず全辺を 1.0（無変形）にしておき、条件に合う辺だけ書き換える
        scale = np.ones(self.edge_i.size)
        scale[(ti == EXPAND) & (tj == EXPAND)] = scale_expand
        scale[(ti == SHRINK) & (tj == SHRINK)] = scale_shrink
        return scale

    def solve(self, node_types, scale_expand, scale_shrink):
        """
        画像ごとの部分（元コードの 2 の目標計算, 5 の b, 6 の b_U, 7 の求解, 8 の統合）。
        Returns: 変形後のノード座標 (N*M, 2)
        """
        if node_types.shape != (self.N, self.M):
            raise ValueError(f"node_types の形 {node_types.shape} がメッシュ {(self.N, self.M)} と違います")
        if not np.array_equal(node_types == FIXED, self.border):
            raise ValueError("FIXED のノードが外周と一致しません（外周固定のみ対応）")

        # ターゲット辺ベクトル t = e * scale（元コードの target_edges と t_vec）
        # edge_vec は (E, 2)、scale は (E,)。scale[:, None] で各辺の x, y 両方に同じ倍率を掛け、
        # ravel で [t0_x, t0_y, t1_x, t1_y, ...] の順に並べる（M_mat の行の並びと同じ）
        scale = self.edge_scales(node_types, scale_expand, scale_shrink)
        t_vec = (self.edge_vec * scale[:, None]).ravel()
        # 右辺 b と境界条件
        b = self.w_target * (self.M_mat.T @ t_vec)
        b_U = b[self.idx_U] - self.A_UC @ self.v_C
        # 求解（分解済みの LU を使う）
        v_U = self.lu.solve(b_U)
        # 結果の統合
        new_coords = np.zeros_like(self.orig_coords)
        new_coords[self.fixed_nodes] = self.orig_coords[self.fixed_nodes]
        new_coords[self.free_nodes] = v_U.reshape(-1, 2)
        return new_coords


# 出力サイズ・重みごとに分解済みの MeshSolver を保持する
# （VisDrone + 800x800 枠なら、内容サイズは実質 800x450 と 800x600 の2種類）
# メモリ注意: 800x600 の LU 分解は約 4.3GB（実測）。保持数の上限を超えたら古いものから捨てる。
# 研究室PC（空き約55GB）を想定して 6 個まで保持する（すべて 800x600 相当でも約26GB）。
# メモリの少ない環境では値を減らし、画像を内容サイズごとに並べて処理すること。
MAX_CACHED_SOLVERS = 6
_SOLVER_CACHE = {}


def get_solver(N, M, w_target=1.0, w_asap=1.0):
    key = (N, M, float(w_target), float(w_asap))
    if key not in _SOLVER_CACHE:
        # 上限を超える分は古いものから捨てる（dict は挿入順を保持する）
        while len(_SOLVER_CACHE) >= MAX_CACHED_SOLVERS:
            _SOLVER_CACHE.pop(next(iter(_SOLVER_CACHE)))
        _SOLVER_CACHE[key] = MeshSolver(N, M, w_target, w_asap)
    return _SOLVER_CACHE[key]


def clear_solver_cache():
    """分解済みの行列をすべて捨ててメモリを空ける。"""
    _SOLVER_CACHE.clear()


def deform_mesh(grid_shape, node_types, scale_expand=1.5, scale_shrink=0.5, w_target=1.0, w_asap=1.0):
    """
    四角形メッシュの領域ベース拡大・縮小変形（元コードと同じ名前・引数・戻り値）
    grid_shape: (N, M) 縦横のノード数
    node_types: (N, M) の行列．0:固定， 1:拡大， 2:縮小
    Returns: (orig_mesh, out_mesh) それぞれ (N, M, 2)

    元コードとの違い:
      - 固定(0)は外周のみ対応（外周以外に 0 があるとエラー）
      - 同じサイズ・重みなら行列の分解を使い回す
      - 既定値は元コードのまま（scale_expand=1.5）。build_forward_map からは 2.0 を渡す
    """
    N, M = grid_shape
    solver = get_solver(N, M, w_target, w_asap)
    new_coords = solver.solve(np.asarray(node_types), scale_expand, scale_shrink)
    return solver.orig_coords.reshape(N, M, 2), new_coords.reshape(N, M, 2)