"""
【描画部分】集計済みの検出ごとの配列（score_iou_match.match_all → select_subset の結果）から図を描く。
集計や対応付けはしない（plot_score_iou.py から呼ぶ）。

  - plot_scatter    : スコア（横）× IoU（縦）の散布図。色 = 相手の GT のサイズ、
                      形 = ○ クラス一致 / × クラス不一致・相手なし
  - plot_histograms : IoU のヒストグラムとスコアのヒストグラムを、それぞれ別のファイルに保存する
凡例は渡されたデータに実際にあるものだけを出すので、「クラス一致のみ」のデータを渡すと
灰色（相手なし）と × は自動で凡例から消える。

置き場所: projects/Edge_poisson_asap/tools/score_iou_figures.py
動作確認: python projects/Edge_poisson_asap/tools/score_iou_figures.py
          （乱数のダミーデータで図を score_iou_figures_test/ に保存する）
"""
import matplotlib
matplotlib.use("Agg")   # 研究環境では plt.show() が使えないので、画面に出さずファイルに保存する
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
from matplotlib.lines import Line2D   # noqa: E402

from score_iou_match import (SIZE_LARGE, SIZE_MEDIUM, SIZE_NAMES, SIZE_NONE,  # noqa: E402
                             SIZE_SMALL)

# サイズ区分ごとの色（相手なしは目立たない灰色）
SIZE_COLORS = {SIZE_SMALL: "#e8590c", SIZE_MEDIUM: "#1971c2",
               SIZE_LARGE: "#2f9e44", SIZE_NONE: "#adb5bd"}
# 点を重ねる順番: 数の多い相手なしを一番下に、小物体を一番上に
DRAW_ORDER = [SIZE_NONE, SIZE_LARGE, SIZE_MEDIUM, SIZE_SMALL]
# ヒストグラムの合計の線の色
TOTAL_COLOR = "#212529"
# 手法比較のヒストグラムで手法ごとに使う色（--preds の順に割り当てる。数が手法の上限）。
# サイズの色（橙・青・緑）と見分けやすい組み合わせにしている
METHOD_COLORS = ["#1971c2", "#e8590c", "#2f9e44", "#9c36b5", "#c2255c"]
# ヒストグラムの区切り（0〜1 を 0.02 刻み）
HIST_BINS = np.linspace(0, 1, 51)


def _draw_points(ax, score, iou, cls_match, color, point_size, alpha):
    """同じ色の点を、クラス一致は ○、不一致・相手なしは × で打つ"""
    for match, marker in [(True, "o"), (False, "x")]:
        mask = cls_match == match
        if not mask.any():
            continue
        # ○ は塗りつぶし（枠線なし）、× は線で描く
        style = {"edgecolors": "none"} if marker == "o" else {"linewidths": 0.6}
        ax.scatter(score[mask], iou[mask], s=point_size, c=color, marker=marker,
                   alpha=alpha, rasterized=True, **style)


def _setup_axes(ax, title):
    """散布図の軸範囲・ラベルをそろえる（横 スコア 0〜1、縦 IoU 0〜1、どちらも線形）"""
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1)     # IoU = 0 の点が下の枠線に隠れないよう、下に少し余白を取る
    ax.set_xlabel("confidence score")
    ax.set_ylabel("IoU with best-overlapping GT (class-agnostic)")
    ax.set_title(title)
    ax.grid(alpha=0.3)


def _legend_handles(m):
    """
    凡例の見本を作る（点は透けて見えにくいので、不透明の見本を別に作る）。
    データに実際にあるサイズ区分と形だけを載せる
    """
    handles = [Line2D([], [], ls="", marker="o", color=SIZE_COLORS[s], label=SIZE_NAMES[s])
               for s in [SIZE_SMALL, SIZE_MEDIUM, SIZE_LARGE, SIZE_NONE] if (m["size"] == s).any()]
    # × がある（クラス不一致・相手なしを含む）ときだけ形の見本を付ける
    if (~m["cls_match"]).any():
        handles += [Line2D([], [], ls="", marker="o", color="k", label="class match"),
                    Line2D([], [], ls="", marker="x", color="k", label="class mismatch / no GT")]
    return handles


def plot_scatter(m, title, path, point_size, alpha):
    """
    スコア × IoU の散布図を保存する（色 = 相手の GT のサイズ）
    m     : 検出ごとの配列の dict{score, iou, size, cls_match}
    title : 図のタイトル（点の数は自動で付け足す）
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for size in DRAW_ORDER:
        mask = m["size"] == size
        _draw_points(ax, m["score"][mask], m["iou"][mask], m["cls_match"][mask],
                     SIZE_COLORS[size], point_size, alpha)
    _setup_axes(ax, f"{title}  (n={len(m['score'])})")
    ax.legend(handles=_legend_handles(m), loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_one_histogram(m, key, xlabel, title, path, log_y):
    """
    1つの値（key = "iou" か "score"）のヒストグラムを、サイズ区分ごとに色分けして重ねて保存する。
    線だけ（step）で描くので重なっても見え、黒い線は全サイズの合計
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    # 合計を先に描き、各サイズの線をその上に重ねる
    ax.hist(m[key], bins=HIST_BINS, histtype="step", linewidth=1.5, color=TOTAL_COLOR,
            label=f"total (n={len(m[key])})")
    for size in [SIZE_SMALL, SIZE_MEDIUM, SIZE_LARGE, SIZE_NONE]:
        values = m[key][m["size"] == size]
        if len(values) == 0:
            continue      # クラス一致のみのデータでは相手なし（no GT）が無いので描かない
        ax.hist(values, bins=HIST_BINS, histtype="step", linewidth=1.2, color=SIZE_COLORS[size],
                label=f"{SIZE_NAMES[size]} (n={len(values)})")
    if log_y:
        # IoU = 0 や低スコアに件数が集中するので、対数にすると少ない帯も見える
        ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count (log)" if log_y else "count")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_histograms(m, title, path_iou, path_score, log_y=True):
    """IoU のヒストグラムとスコアのヒストグラム（サイズで色分け）を、それぞれ path_iou / path_score に保存する"""
    _plot_one_histogram(m, "iou", "IoU", title, path_iou, log_y)
    _plot_one_histogram(m, "score", "confidence score", title, path_score, log_y)


def plot_compare_histogram(datas, names, key, xlabel, title, path, log_y=True):
    """
    複数の手法の同じ値（key = "iou" か "score"）のヒストグラムを、手法ごとに色を変えて1枚に重ねて保存する。
    サイズでの色分けはしない。線だけ（step）で描くので重なっても見える。
    datas : 手法ごとの検出の配列の dict のリスト（select_subset の結果）
    names : 手法名のリスト（datas と同じ順）。凡例に件数と一緒に出す
    """
    if len(datas) > len(METHOD_COLORS):
        raise ValueError(f"手法は {len(METHOD_COLORS)} 個までです（色が足りません）")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, name, color in zip(datas, names, METHOD_COLORS):
        ax.hist(m[key], bins=HIST_BINS, histtype="step", linewidth=1.5, color=color,
                label=f"{name} (n={len(m[key])})")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count (log)" if log_y else "count")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    # 動作確認: 乱数のダミーデータで3つの図を保存する
    import os
    out_dir = "score_iou_figures_test"
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    n = 3000
    dummy = {"score": rng.uniform(0.01, 1, n),
             "iou": np.where(rng.uniform(size=n) < 0.3, 0.0, rng.uniform(0, 1, n)),
             "size": rng.choice([SIZE_SMALL, SIZE_MEDIUM, SIZE_LARGE], n)}
    dummy["size"] = np.where(dummy["iou"] > 0, dummy["size"], SIZE_NONE)   # IoU = 0 は相手なし
    dummy["cls_match"] = (dummy["iou"] > 0) & (rng.uniform(size=n) < 0.8)
    plot_scatter(dummy, "dummy", os.path.join(out_dir, "scatter.png"), 4.0, 0.2)
    plot_histograms(dummy, "dummy", os.path.join(out_dir, "hist_iou.png"),
                    os.path.join(out_dir, "hist_score.png"))
    # 比較: IoU を少し下げた「手法B」を作り、2手法を重ねる
    dummy_b = {**dummy, "iou": dummy["iou"] * 0.9}
    plot_compare_histogram([dummy, dummy_b], ["A", "B"], "iou", "IoU", "dummy compare",
                           os.path.join(out_dir, "compare_hist_iou.png"))
    print(f"保存先: {out_dir}")