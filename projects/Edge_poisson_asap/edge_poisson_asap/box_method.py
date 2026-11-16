"""
ボックスの作り方（mode, n_per_side）の指定を、コマンドライン引数・settings.json・設定ファイルで共通に扱う

  - add_box_args       : argparse に --bbox-mode と --bbox-n-per-side を加える
  - parse_n_per_side   : --bbox-n-per-side の文字列（"3" や "all"）を値にする
  - read_box_method    : make_deformed_dataset.py が書いた settings.json から、正解を作ったときの方式を読む
  - resolve_box_method : 実際に使う方式を決める（指定 > settings.json > 既定値 の順）。
                         指定と settings.json が違えば警告する（正解と検出で作り方がずれるため）

なぜ settings.json を読むのか
  正解ボックスは make_deformed_dataset.py で forward_bbox（順写像）を使って作り、
  検出結果は評価・描画のときに inverse_bbox（逆写像）で元画像の座標に戻す。
  2つの方式がそろっていないと、方式の違いがそのまま位置ずれとして評価に入る。
  そこで、データ作成時の方式を settings.json に残し、逆変換の側は既定でそれを使う。

動作確認（projects/Edge_poisson_asap/ で実行）:
    python -m edge_poisson_asap.box_method
  辺の長さごとの点の数と、写像が恒等（変形なし）のときに枠が変わらないことを表示する。
"""
import json
import os

from edge_poisson_asap.box_transform import BOX_MODES, check_n_per_side

# settings.json に方式の記録がない古いデータ（この改修より前に作ったもの）に使う値。
# 改修前の forward_bbox の既定値と同じにしてある
DEFAULT_MODE = "mean"
DEFAULT_N_PER_SIDE = 3
# settings.json での名前（make_deformed_dataset.py の引数名 --bbox-mode, --bbox-n-per-side と同じ）
KEY_MODE = "bbox_mode"
KEY_N = "bbox_n_per_side"


def parse_n_per_side(text):
    """argparse の type 用: "all" はそのまま、数字は int にして、2 以上か確かめる"""
    try:
        return check_n_per_side(text)
    except ValueError as e:
        import argparse
        raise argparse.ArgumentTypeError(str(e))


def add_box_args(parser, from_settings=True):
    """
    argparse に --bbox-mode と --bbox-n-per-side を加える。
    from_settings: True なら既定値を None にして、resolve_box_method で settings.json の値を使う
                   （逆変換の側）。False なら既定値を DEFAULT_* にする（正解を作る側）
    """
    note = "省略すると --warp-dir の settings.json の値" if from_settings else f"既定 {DEFAULT_MODE}"
    parser.add_argument("--bbox-mode", choices=BOX_MODES, default=None if from_settings else DEFAULT_MODE,
                        help=f"ボックスの作り方（corner2 / mean / minmax）。{note}")
    note = "省略すると settings.json の値" if from_settings else f"既定 {DEFAULT_N_PER_SIDE}"
    parser.add_argument("--bbox-n-per-side", type=parse_n_per_side,
                        default=None if from_settings else DEFAULT_N_PER_SIDE,
                        help="1本の辺に置く点の数（2 以上の整数か all。辺の長さで決まる上限を超えた分は減らす。"
                             f"corner2 では使わない）。{note}")


def read_box_method(warp_dir):
    """
    {warp_dir}/settings.json から、正解を作ったときの (mode, n_per_side) を読む。
    ファイルがない・方式の記録がない（古いデータ）ときは、ないものを None で返す。
    """
    path = os.path.join(warp_dir, "settings.json") if warp_dir else None
    if path is None or not os.path.isfile(path):
        return None, None
    with open(path) as f:
        settings = json.load(f)
    mode = settings.get(KEY_MODE)                  # 正解を作ったときのボックスの作り方
    n = settings.get(KEY_N)                        # 正解を作ったときの 1 辺の点の数
    return mode, (check_n_per_side(n) if n is not None else None)


def resolve_box_method(warp_dir, mode=None, n_per_side=None, verbose=True):
    """
    実際に使う (mode, n_per_side) を決める。
      優先順: 引数で指定した値 > settings.json の値 > DEFAULT_MODE, DEFAULT_N_PER_SIDE
    warp_dir  : make_deformed_dataset.py の --out-dir（settings.json がある所）。None なら読まない
    mode, n_per_side: 指定された値（None なら指定なし）
    verbose   : True なら、使う方式と警告を表示する
    """
    saved_mode, saved_n = read_box_method(warp_dir)
    use_mode = mode or saved_mode or DEFAULT_MODE   # 実際に使う作り方
    use_n = check_n_per_side(n_per_side if n_per_side is not None
                             else saved_n if saved_n is not None else DEFAULT_N_PER_SIDE)
    if verbose:
        if saved_mode is None:
            print(f"[注意] settings.json に正解を作ったときの方式の記録がありません（{warp_dir}）。"
                  "作成時と同じ方式か確認してください")
        else:
            # 方式が違うと、正解と検出で枠の作り方がずれる。corner2 では点の数は使わないので比べない
            differ = use_mode != saved_mode or (use_mode != "corner2" and use_n != saved_n)
            if differ:
                print(f"[警告] 正解を作ったときの方式（mode={saved_mode}, n_per_side={saved_n}）と"
                      f"違う方式で逆変換します（mode={use_mode}, n_per_side={use_n}）")
        print(f"ボックスの作り方: mode={use_mode}, n_per_side={use_n}"
              f"{'（corner2 なので点の数は使わない）' if use_mode == 'corner2' else ''}")
    return use_mode, use_n


if __name__ == "__main__":
    # 動作確認: 辺の長さと点の数の上限、恒等写像での往復
    import numpy as np
    from types import SimpleNamespace

    from edge_poisson_asap.box_transform import _side_counts, forward_bbox, inverse_bbox

    boxes = np.array([[10.0, 20.0, 0.6, 0.6],     # 1 画素未満（角だけになる）
                      [10.0, 20.0, 3.0, 5.0],     # 小さい物体
                      [10.0, 20.0, 40.0, 25.5]])  # 大きめの物体（高さが小数）
    for n in (2, 3, 5, 10, "all"):
        counts = _side_counts(boxes, n)            # (K, 4) 左・右・上・下の辺の点の数
        print(f"n_per_side={n!s:>3}: 左右の辺 {counts[:, 0].tolist()}, 上下の辺 {counts[:, 2].tolist()}")

    # 変形のない写像（ノード数 = 画素数、dst = 元の位置）では、どの方式でも枠は変わらないはず
    H, W = 60, 80                                   # 小さな画像
    yy, xx = np.mgrid[0:H, 0:W].astype(float)       # 画素の中心の座標
    fmap = SimpleNamespace(in_size=(H, W), content_size=(H, W), dst=np.stack([xx, yy], axis=-1))
    for mode in BOX_MODES:
        for n in (3, "all"):
            fwd = forward_bbox(boxes, fmap, mode, n)
            back = inverse_bbox(fwd, xx, yy, mode, n)
            print(f"{mode:7s} n={n!s:>3}: 往復の最大誤差 {np.abs(back - boxes).max():.2e}")