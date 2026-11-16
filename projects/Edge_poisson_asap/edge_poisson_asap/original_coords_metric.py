"""
mmdet の評価（val / test）を、元画像の座標・元画像の物体のサイズ区分で行うための評価クラス

非一様縮小のデータで学習した検出器は、変形後の画像の座標で検出結果を出す。
標準の CocoMetric のままだと、変形後の正解（拡大された物体は medium/large に移る）で
評価され、一様縮小と比べられない。このクラスは、検出結果を受け取るたびに
ノード位置（make_deformed_dataset.py --save-mesh）から作った逆写像で元画像の座標に戻し、
元の正解 json（my_val.json）で評価する。
これにより、学習中の val、best の選択、tools/test.py の結果がすべて
「元画像の座標・元のサイズ区分」になり、一様縮小の CocoMetric と同じ基準で比べられる。

設定ファイルでの使い方（yolox_tiny_visdrone_poisson_asap.py）:
    custom_imports = dict(imports=['edge_poisson_asap.original_coords_metric'],
                          allow_failed_imports=False)
    val_evaluator = dict(type='OriginalCoordsCocoMetric',
                         ann_file='data_ann/VisDrone/my_val.json',            # 元の正解
                         mesh_dir='data_ann/VisDrone_poisson_asap/val/meshes',
                         bbox_mode=None,          # 省略可。None なら mesh_dir の1つ上の settings.json の値
                         bbox_n_per_side=None,    # 省略可。2 以上の整数か 'all'
                         metric='bbox')
  custom_imports で読み込めるよう、projects/Edge_poisson_asap/ を PYTHONPATH に加えて実行する:
    PYTHONPATH=projects/Edge_poisson_asap:$PYTHONPATH python tools/train.py ...
  方式を試しに変えるときは、設定ファイルを書き換えずに --cfg-options でも指定できる:
    --cfg-options val_evaluator.bbox_mode=minmax val_evaluator.bbox_n_per_side=all

ボックスの扱いは標準の CocoMetric にそろえ、既定では元画像の範囲への切り詰めをしない
（一様縮小の CocoMetric も切り詰めないため）。
"""
import os
from types import SimpleNamespace

import numpy as np
from mmdet.evaluation.metrics import CocoMetric
from mmdet.registry import METRICS

from .box_method import resolve_box_method
from .box_transform import inverse_bbox
from .poisson_asap_remap import build_inverse_map


@METRICS.register_module()
class OriginalCoordsCocoMetric(CocoMetric):
    """
    mesh_dir : ノード位置（{元画像の名前}.npz）のフォルダ
    clip     : True なら戻したボックスを元画像の範囲に切り詰める（既定 False = CocoMetric と同じ）
    cache_maps: True なら画像ごとの逆写像をメモリに保持して、2回目以降の評価を速くする
                （val 548 枚で約 2GB）
    bbox_mode, bbox_n_per_side: 検出を元の座標へ戻すときのボックスの作り方（box_transform.py を参照）。
                None なら、mesh_dir の1つ上（make_deformed_dataset.py の --out-dir）の settings.json の値
    その他の引数（ann_file, metric など）は CocoMetric と同じ。ann_file には元の正解 json を渡す。
    """

    def __init__(self, mesh_dir, clip=False, cache_maps=True, bbox_mode=None, bbox_n_per_side=None,
                 **kwargs):
        super().__init__(**kwargs)
        if self._coco_api is None:
            raise ValueError("OriginalCoordsCocoMetric には ann_file（元の正解 json）が必要です")
        self.mesh_dir = mesh_dir
        self.clip = clip
        self.cache_maps = cache_maps
        self._map_cache = {}                       # 画像 id -> (map_x, map_y, 元画像の大きさ)
        # 正解を作ったときと同じ方式で戻すため、settings.json（meshes/ の1つ上）を見て決める
        warp_dir = os.path.dirname(os.path.normpath(mesh_dir))
        self.bbox_mode, self.bbox_n_per_side = resolve_box_method(warp_dir, bbox_mode, bbox_n_per_side)

    def _inverse_maps(self, img_id):
        """画像 id に対応する逆写像 (map_x, map_y) と元画像の大きさを返す"""
        if img_id in self._map_cache:
            return self._map_cache[img_id]
        # ノード位置のファイル名は、元画像のファイル名から拡張子を除いたもの
        # （画像 id は変形の前後で同じなので、元の正解 json から名前を引ける）
        stem = os.path.splitext(self._coco_api.imgs[img_id]["file_name"])[0]
        z = np.load(os.path.join(self.mesh_dir, stem + ".npz"))
        in_size = tuple(int(v) for v in z["in_size"])   # 元画像の大きさ (高さ, 幅)
        if "map_x" in z.files:
            # データ作成時に逆写像も保存してあれば、それを使う（計算不要で速い）
            map_x, map_y = z["map_x"], z["map_y"]
        else:
            fmap = SimpleNamespace(in_size=in_size,
                                   content_size=tuple(int(v) for v in z["content_size"]),
                                   dst=z["dst"].astype(np.float64))
            map_x, map_y = build_inverse_map(fmap)
        out = (map_x, map_y, in_size)
        if self.cache_maps:
            self._map_cache[img_id] = out
        return out

    def process(self, data_batch, data_samples):
        """
        標準の処理（検出結果と画像情報を self.results に追加）を行ったあと、
        今回追加した分のボックスを元画像の座標に戻し、画像の大きさも元画像のものにする。
        """
        n_before = len(self.results)               # 今回の追加分を見分けるための、追加前の数
        super().process(data_batch, data_samples)
        for gt, result in self.results[n_before:]:
            map_x, map_y, (in_h, in_w) = self._inverse_maps(result["img_id"])
            boxes = result["bboxes"]                     # [x1, y1, x2, y2]（変形後の座標）
            if len(boxes):
                xywh = np.concatenate([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], axis=1)
                # box を元画像の座標へ（方式は __init__ で決めたもの）
                xywh = inverse_bbox(xywh, map_x, map_y, self.bbox_mode, self.bbox_n_per_side)
                new = np.concatenate([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], axis=1)
                if self.clip:
                    new[:, [0, 2]] = np.clip(new[:, [0, 2]], 0, in_w)
                    new[:, [1, 3]] = np.clip(new[:, [1, 3]], 0, in_h)
                result["bboxes"] = new.astype(boxes.dtype)
            gt["width"], gt["height"] = in_w, in_h