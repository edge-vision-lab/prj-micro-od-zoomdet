# VisDrone2019-DET（10 クラス）の dataset 設定。
# ZoomDet 環境のディレクトリ構成（zoomdet-setup-record.md の B-3）をそのまま使う:
#   data/VisDrone/
#     VisDrone2019-DET-train/  images/  my_train.json
#     VisDrone2019-DET-val/    images/  my_val.json
# json は tools/dataset_converters/visdrone2coco.py で作る。
#
# パイプラインの Resize は ZoomDet の Uniform 実験と同じ (1333, 800)。
# 素の mmdet で Uniform（AP 22.4）が再現できるかの確認に使う。
# MCU 向けの軽量検出器では、検出器側の設定でパイプラインを上書きする。
dataset_type = 'CocoDataset'
data_root = 'data/VisDrone/'
metainfo = dict(
    classes=('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
             'tricycle', 'awning-tricycle', 'bus', 'motor'))
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='VisDrone2019-DET-train/my_train.json',
        data_prefix=dict(img='VisDrone2019-DET-train/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='VisDrone2019-DET-val/my_val.json',
        data_prefix=dict(img='VisDrone2019-DET-val/images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'VisDrone2019-DET-val/my_val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator
