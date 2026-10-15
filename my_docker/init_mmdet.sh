#!/bin/bash

# エラーメッセージ等を出さないようにしてクリーンに実行
echo "Restoring MMDetection environment..."

cd $HOME/workspace
NUMPY_VER=$(python -c "import numpy; print(numpy.__version__)")
pip install -r requirements/runtime.txt "numpy==$NUMPY_VER" > /dev/null 2>&1
python setup.py develop > /dev/null 2>&1

echo "Done!"

# 最後に通常のbashプロンプトへ引き継ぐ（重要）
exec /usr/bin/bash
