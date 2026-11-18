# ===========================================================================
# YOLOX-tiny / SeaDronesSee ODv2 / 一様縮小（比較対象）
#
# 実行（workspace/ で）:
#   python tools/train.py projects/Edge_poisson_asap/configs/yolox_tiny_sds_uniform.py \
#       --auto-scale-lr --work-dir work_dirs/yolox_tiny_sds_uniform
#   --auto-scale-lr: 学習率の基準は「8GPU x 8枚 = 64枚」なので、実際のバッチの大きさに
#                    合わせて学習率を自動で比例させる（1GPU x 8枚なら 1/8 になる）
#
# yolox_tiny_visdrone_uniform.py からの変更点（それ以外は同じ）
#   - データセット: VisDrone -> SeaDronesSee ODv2（Kaggle の compressed 版）
#       クラス数 10 -> 5（swimmer, boat, jetski, life_saving_appliances, buoy）
#       アノテーションは tools/dataset_converters/seadronessee2coco.py で作った
#       data_ann/SeaDronesSee/my_{train,val}.json
#       （test の正解は非公開なので、VisDrone と同じく公式 val で評価する）
#   - 学習時間の見込みを書き換え（SDS は 4K 級の画像が約9割で、読み込みと縮小が重い）
#   - データ読み込みを並列にした（num_workers=4）。並列なしでは 1 イテレーション 1.16 秒の
#     うち 1.04 秒が画像の読み込み待ちで、50 エポックに約18時間かかる見込みだったため。
#     前提: docker compose の override で shm_size を増やしてコンテナを作り直してあること
#     （共有メモリ 64MB のままだと並列読み込みで止まる）
#
# mmdet 標準の yolox_tiny_8xb8-300e_coco.py からの変更点（VisDrone 版と共通）
#   - クラス数 80 -> 5。COCO で学習済みの重みから開始する
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
#   - チェックポイントは 2 エポックごと・最新 5 個を保持し、val の mAP が最も高いもの（best）も残す
# ===========================================================================
_base_ = '../../../configs/yolox/yolox_tiny_8xb8-300e_coco.py'

# 名前は公式 json のカテゴリ名と一致させる（mmdet は名前から category_id を引く）
classes = ('swimmer', 'boat', 'jetski', 'life_saving_appliances', 'buoy')
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
train_ann = 'data_ann/SeaDronesSee/my_train.json'
train_img = 'data/SeaDronesSee/images/train/'
val_ann = 'data_ann/SeaDronesSee/my_val.json'
val_img = 'data/SeaDronesSee/images/val/'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='YOLOXHSVRandomAug'),                         # 色の変化
    dict(type='RandomFlip', prob=0.5),                      # 左右反転
    dict(type='Resize', scale=img_scale, keep_ratio=True),  # 枠に縦横比を保って収める
    # 縮小後に幅か高さが 1px 未満になった正解は学習から外す
    # （SDS は縮小率が 0.15〜0.2 程度と強いので、VisDrone より外れる正解が多い可能性がある）
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

# データ読み込みの並列プロセス数。4K 画像の JPEG 展開と縮小が重いので並列にする。
# 前提: コンテナの共有メモリ（/dev/shm）を override の shm_size で増やしてあること。
# 増やしていない（64MB のまま）なら 0 にする（このとき persistent_workers は False になる）
num_workers = 8

train_dataloader = dict(
    batch_size=8,
    num_workers=num_workers,
    # 並列プロセスをエポックをまたいで使い回す（num_workers=0 のときは False でないとエラー）
    persistent_workers=num_workers > 0,
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
    num_workers=num_workers,
    persistent_workers=num_workers > 0,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=val_ann,
        data_prefix=dict(img=val_img),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', ann_file=val_ann, metric='bbox', classwise=True)
test_evaluator = val_evaluator

# --- 学習スケジュール -----------------------------------------------------
# COCO で学習済みの重みから始めるので、標準（ゼロから 300 エポック）より短くする。
# 学習時間: 並列なし（num_workers=0）では 1 イテレーション約1.16秒（うち読み込み待ち約1.04秒）、
# 1 エポック 1117 イテレーションで、50 エポックに約18時間の見込みだった。
# GPU での計算は 1 イテレーション約0.13秒なので、読み込みが追いつけば大きく短くなるはず。
# 並列にした後の時間は、最初のログに出る残り時間（eta）と data_time で確認すること
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
default_hooks = dict(
    checkpoint=dict(interval=2, max_keep_ckpts=5,
                    save_best='coco/bbox_mAP', rule='greater'))
custom_hooks = [
    dict(type='SyncNormHook', priority=48),
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0001,
         update_buffers=True, priority=49),
]