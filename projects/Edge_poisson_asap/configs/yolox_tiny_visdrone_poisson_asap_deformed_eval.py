"""
切り分け用の設定: 新手法の検出結果を「変形後の座標のまま」json に出す

元の設定（yolox_tiny_visdrone_poisson_asap.py）を引き継ぎ、評価クラスだけを
元座標に戻す OriginalCoordsCocoMetric から標準の CocoMetric に置き換える。
この実行で画面に出る AP は変形後の座標・変形後のサイズ区分なので、比較には使わない。
欲しいのは outfile_prefix に出る results.bbox.json（変形後の座標）だけ。
"""
_base_ = './yolox_tiny_visdrone_poisson_asap.py'

test_evaluator = dict(
    _delete_=True,   # 元の評価クラスの設定（mesh_dir など）を引き継がない
    type='CocoMetric',
    ann_file='data_ann/VisDrone_poisson_asap/val/annotations.json',   # 変形後の座標の正解
    metric='bbox',
    outfile_prefix='work_dirs/eval_visdrone_poisson_asap_e50_deformed/results')