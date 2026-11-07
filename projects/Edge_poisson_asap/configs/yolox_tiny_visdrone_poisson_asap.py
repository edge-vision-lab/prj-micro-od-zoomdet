# ===========================================================================
# YOLOX-tiny / VisDrone / 非一様縮小（ポアソン＋ASAP）
#
# 一様縮小の設定（yolox_tiny_visdrone_uniform.py）をそのまま使い、データの場所と評価方法だけ変える。
# データは tools/make_deformed_dataset.py で事前に作ったもの:
#   画像はすでに枠（800x800）に収まる大きさなので、Resize は何もしない（倍率 1）
#   正解ボックスも変形後の座標に変換済み（annotations.json）
#
# 評価（学習中の val と tools/test.py）は OriginalCoordsCocoMetric で行う:
#   検出結果を元画像の座標に戻し、元の正解（my_val.json）で評価するので、
#   small/medium/large の区分も含めて一様縮小の CocoMetric と同じ基準になる。
#   best の選択もこの値で行われる。
#   val の meshes は make_deformed_dataset.py --save-mesh で作ったもの（逆写像を含むと速い）
#
# 実行（workspace/ で）。評価クラスを読み込むため PYTHONPATH に projects/Edge_poisson_asap を加える:
#   PYTHONPATH=projects/Edge_poisson_asap:$PYTHONPATH python tools/train.py \
#       projects/Edge_poisson_asap/configs/yolox_tiny_visdrone_poisson_asap.py \
#       --auto-scale-lr --work-dir work_dirs/yolox_tiny_visdrone_poisson_asap_e50
# ===========================================================================
_base_ = './yolox_tiny_visdrone_uniform.py'

# 評価クラス OriginalCoordsCocoMetric を登録する（edge_poisson_asap/original_coords_metric.py）
custom_imports = dict(imports=['edge_poisson_asap.original_coords_metric'],
                      allow_failed_imports=False)

data_dir = 'data_ann/VisDrone_poisson_asap/'   # make_deformed_dataset.py の --out-dir の親
train_ann = data_dir + 'train/annotations.json'
train_img = data_dir + 'train/images/'
val_ann = data_dir + 'val/annotations.json'    # 画像の読み込み用（変形後の座標の正解）
val_img = data_dir + 'val/images/'
orig_val_ann = 'data_ann/VisDrone/my_val.json'  # 評価用（元画像の座標の正解）

train_dataloader = dict(dataset=dict(ann_file=train_ann, data_prefix=dict(img=train_img)))
val_dataloader = dict(dataset=dict(ann_file=val_ann, data_prefix=dict(img=val_img)))
test_dataloader = val_dataloader
val_evaluator = dict(
    type='OriginalCoordsCocoMetric',
    ann_file=orig_val_ann,                  # 元の正解で評価する（サイズ区分も元のまま）
    mesh_dir=data_dir + 'val/meshes',       # 検出結果を元画像の座標に戻すためのノード位置・逆写像
    metric='bbox')
test_evaluator = val_evaluator