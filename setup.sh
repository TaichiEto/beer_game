#!/bin/bash

echo "Beer Game の環境をセットアップします..."

# 仮想環境の作成（存在しない場合）
if [ ! -d "venv" ]; then
    echo "仮想環境を作成しています..."
    python3 -m venv venv
fi

# 仮想環境のアクティベート
source venv/bin/activate

# 依存関係のインストール
echo "必要なパッケージをインストールしています..."
pip install --upgrade pip
pip install -r requirements.txt

echo "セットアップが完了しました！"
echo "以下のコマンドで環境をアクティベートしてください："
echo "source venv/bin/activate"
