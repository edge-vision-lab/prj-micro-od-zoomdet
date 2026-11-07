# ===========================================================================
# YOLOX-tiny / VisDrone / 一様縮小（比較対象）
#
# 実行（workspace/ で）:
#   python tools/train.py projects/Edge_poisson_asap/configs/yolox_tiny_visdrone_uniform.py \
#       --auto-scale-lr --work-dir work_dirs/yolox_tiny_visdrone_uniform
#   --auto-scale-lr: 学習率の基準は「8GPU x 8枚 = 64枚」なので、実際のバッチの大きさに
#                    合わせて学習率を自動で比例させる（1GPU x 8枚なら 1/8 になる）
#
# mmdet 標準の yolox_tiny_8xb8-300e_coco.py からの変更点
#   - クラス数 80 -> 10（VisDrone）。COCO で学習済みの重みから開始する
#   - 入力: 800x800 の枠に縦横比を保って縮小（Resize）し、32 の倍数へのパディングは
#           data_preprocessor が行う（正方形へのパディングはしない）
#   - データ拡張: Mosaic, RandomAffine, MixUp, BatchSyncRandomResize（入力サイズのランダム変更）
#                を使わない。色の変化（YOLOXHSVRandomAug）と左右反転は残す
#   - YOLOXModeSwitchHook を外した（Mosaic などを最後の15エポックで切るための hook で、
#     Mosaic を使わない今回は不要。また MultiImageMixDataset 以外ではエラーになる）
#     この hook は最後の15エポックで L1 損失も加えるが、それも行わない
#   - 学習スケジュール: 標準の 300 エポック（ゼロから学習する前提）をやめ、学習済みの重みから
#     50 エポック学習する。ウォームアップ 3 エポック -> 最後まで cos 型で下げる
#     （標準の「最後の15エポックは学習率一定」は、外した YOLOXModeSwitchHook 用なのでなくした）
#   - データ読み込みの並列プロセスを使わない（コンテナの共有メモリが 64MB しかなく、
#     並列にすると共有メモリ不足で止まるため）
#   - チェックポイントは 2 エポックごと・最新 5 個を保持し、val の mAP が最も高いもの（best）も残す
# ===========================================================================
_base_ = '../../../configs/yolox/yolox_tiny_8xb8-300e_coco.py'

classes = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle',
           'awning-tricycle', 'bus', 'motor')
metainfo = dict(classes=classes)
img_scale = (800, 800)   # 縮小の枠 (幅, 高さ)。非一様縮小のデータ作成時の --box と合わせる

# COCO で学習済みの YOLOX-tiny（mmdet 3.3.0 の configs/yolox/metafile.yml に載っている重み）
load_from = ('https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/'
             'yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth')

model = dict(
    data_preprocessor=dict(
        pad_size_divisor=32,      # バッチ内の画像を 32 の倍数の大きさへパディング
        batch_augments=None),     # BatchSyncRandomResize を使わない
    bbox_head=dict(num_classes=len(classes)))

# --- データ ---------------------------------------------------------------
dataset_type = 'CocoDataset'
data_root = ''   # 以下のパスは workspace/ からの相対パス
train_ann = 'data_ann/VisDrone/my_train.json'
train_img = 'data/VisDrone/VisDrone2019-DET-train/images/'
val_ann = 'data_ann/VisDrone/my_val.json'
val_img = 'data/VisDrone/VisDrone2019-DET-val/images/'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='YOLOXHSVRandomAug'),                         # 色の変化
    dict(type='RandomFlip', prob=0.5),                      # 左右反転
    dict(type='Resize', scale=img_scale, keep_ratio=True),  # 枠に縦横比を保って収める
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# 共有メモリ（/dev/shm）が 64MB しかないため、並列読み込み（num_workers > 0）を使わない。
# 共有メモリを増やしたら num_workers=4, persistent_workers=True に戻すと速くなる
train_dataloader = dict(
    batch_size=8,
    num_workers=0,
    persistent_workers=False,   # num_workers=0 のときは False にしないとエラーになる
    dataset=dict(
        _delete_=True,            # 標準設定の MultiImageMixDataset（Mosaic 用）を使わない
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann,
        data_prefix=dict(img=train_img),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=val_ann,
        data_prefix=dict(img=val_img),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', ann_file=val_ann, metric='bbox',classwise=True)
test_evaluator = val_evaluator

# --- 学習スケジュール -----------------------------------------------------
# COCO で学習済みの重みから始めるので、標準（ゼロから 300 エポック）より短くする。
# 1エポック約4.5分（RTX 3090, 並列読み込みなし）で、50 エポックは約4時間の見込み
max_epochs = 50
warmup_epochs = 3         # 最初に学習率を 0 から徐々に上げるエポック数
val_interval = 10         # 何エポックごとに val で評価するか
base_lr = 0.01            # 64枚のバッチでの値。--auto-scale-lr で実際のバッチに比例させる

train_cfg = dict(max_epochs=max_epochs, val_interval=val_interval)
optim_wrapper = dict(optimizer=dict(lr=base_lr))
param_scheduler = [
    # 最初の warmup_epochs で学習率を 0 から 2次関数的に上げる（1回ごとに更新）
    dict(type='mmdet.QuadraticWarmupLR', by_epoch=True, begin=0, end=warmup_epochs,
         convert_to_iter_based=True),
    # その後、最後まで cos 型で最大値の 5% まで下げる。
    # T_max は「begin から数えて何エポックで最小値に届くか」（mmengine の数え方）なので、
    # 残りのエポック数と同じにすると、ちょうど最後のエポックで最小値に届く
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.05,
         begin=warmup_epochs, end=max_epochs, T_max=max_epochs - warmup_epochs,
         by_epoch=True, convert_to_iter_based=True),
]

# チェックポイント: 2 エポックごと・最新 5 個に加えて、val の mAP が最も高かったものを
# best_coco_bbox_mAP_epoch_{n}.pth として残す（最新 5 個の数には含まれず、消されない）。
# 評価は val_interval ごとなので、best の候補は 10, 20, ..., 50 エポック目の5つ。
# 注意: 非一様縮小では val の評価が「変形後の座標」なので、best もその値で選ばれる
default_hooks = dict(
    checkpoint=dict(interval=2, max_keep_ckpts=5,
                    save_best='coco/bbox_mAP', rule='greater'))
custom_hooks = [
    dict(type='SyncNormHook', priority=48),
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0001,
         update_buffers=True, priority=49),
]