# projects/Edge_poisson_asap/configs/yolox_tiny_visdrone_poisson_asap_mean2.py
# 既存の非一様縮小の設定を読み込み、データの場所だけを新しいものに変える
_base_ = './yolox_tiny_visdrone_poisson_asap.py'

# 新しく作ったデータのフォルダ（workspace からの相対パス）
new_root = 'data_ann/VisDrone_poisson_asap_mean_2/'

# 学習データ：正解の json と画像フォルダを差し替える
train_dataloader = dict(dataset=dict(
    ann_file=new_root + 'train/annotations.json',
    data_prefix=dict(img=new_root + 'train/images/')))

# 検証データ：同じく差し替える
val_dataloader = dict(dataset=dict(
    ann_file=new_root + 'val/annotations.json',
    data_prefix=dict(img=new_root + 'val/images/')))
test_dataloader = val_dataloader   # テストも検証と同じデータ