# ビールゲーム × DQN エージェント

## 概要
強化学習（DQN）を用いて **ビールゲーム（Beer Game）** における最適な発注戦略を学習するプログラムです。
ビールゲームはサプライチェーンの管理を模擬するシミュレーションであり、在庫管理・発注量の決定を強化学習エージェントに学習させることで、
コスト最小化・利益最大化・環境負荷削減などの目的を達成しようと試みます。

## 特徴
✅ **ビールゲームのシミュレーション環境を構築**  
✅ **PyTorchを用いたDQNエージェントを実装**  
✅ **ターゲットネットワークを導入し学習の安定化**  
✅ **報酬関数をカスタマイズ可能（コスト最小化・利益最大化・環境負荷削減）**  
✅ **学習曲線の可視化によるパフォーマンスの評価**  
✅ **訓練後の自動テストによる性能評価**

## 環境構築
### 🔹　リポジトリのクローン
```bash
git clone https://github.com/TaichiEto/beer_game
cd beer_game
```

### 🔹 自動セットアップ（推奨）
セットアップスクリプトを使用すると、互換性のあるバージョンで仮想環境が自動的に構築されます：
```bash
# スクリプトに実行権限を付与
chmod +x setup.sh

# セットアップスクリプトを実行
./setup.sh

# 仮想環境を有効化
source venv/bin/activate
```

### 🔹 手動セットアップ
自分で環境を構築する場合は、以下のコマンドを実行します：
```bash
# 仮想環境の作成
python3 -m venv venv
source venv/bin/activate

# 必要なライブラリをインストール
pip install -r requirements.txt
```

### 🔹 実行方法
以下のコマンドで学習を開始できます：
```bash
python beer_game_dqn.py
```

## 使い方
### 🔹 設定（`config`(スクリプト内に実装) を変更）
```python
config = {
    "time_unit": "week",  # "week", "day", "month"
    "goal": "profit_max",  # "cost_min", "profit_max", "env_min", "weighted"
    "reward_weights": {"cost_min": 0.5, "profit_max": 0.3, "env_min": 0.2},
    "batch_size": 32,     # 学習時のバッチサイズ
    "num_episodes": 500   # 学習エピソード数
}
```

- **`time_unit`**: シミュレーションの時間単位を設定（週・日・月）
- **`goal`**: 学習の目的（コスト最小化、利益最大化、環境負荷削減、または加重平均）
- **`reward_weights`**: 加重平均モードでの目的別の重みを設定
- **`batch_size`**: ミニバッチサイズを設定
- **`num_episodes`**: 学習するエピソード総数を設定

## 出力
1. **学習の進捗（エピソードごとの報酬）**
   ```
   Episode 0/500: Total Reward: -1200.50, Steps: 50, Epsilon: 0.9950
   Episode 10/500: Total Reward: -980.30, Steps: 50, Epsilon: 0.9512
   ...
   ```
2. **学習曲線の可視化**
   - `matplotlib` を使用して学習の進行状況をプロット
   - `output/YYYYMMDD_HHMMSS/reward_plot.png` に保存
3. **テスト後の平均報酬**
   ```
   平均テスト報酬: -850.20
   ```
   - `output/YYYYMMDD_HHMMSS/test_result.txt` に保存

## 人間プレイヤーとの比較機能
### 🔹 追加機能
🔹 **今後実装予定: GUI または CLI で手動操作可能**  
🔹 **今後実装予定: AI と人間のプレイ結果（在庫変動・コスト・報酬）を比較**  
🔹 **今後実装予定: AI vs 人間の評価指標（コスト最小化、利益最大化、安定性 など）を導入**  

## 今後の改良点
- **報酬関数の最適化**
- **強化学習アルゴリズムの追加（PPO、DDPGなど）**
- **プロジェクトマネジメントモデルへの拡張**
- **学習のハイパーパラメータ調整機能**

## 環境負荷低減（Environmental Impact Reduction）
現在の実装では、環境負荷（env_impact）は「発注量（order）」に比例する負荷コストとなっています。

🔹 環境負荷の計算方法
```python
env_impact = self.order * 0.2  # 発注量に応じた環境負荷
```
- 発注量（order）が増えると、輸送回数や資源使用量が増加 → 環境負荷が高くなる
- 逆に、発注量を抑えると環境負荷は減少する

🔹 現在の環境負荷低減の仕組み
- 目的が「環境負荷低減（env_min）」の場合、報酬は以下のようになる
```python
reward = -env_impact  # 発注量が少ないほど報酬が高くなる
```
  → エージェントは「発注量を少なくする」＝「輸送・生産の負担を減らす」ことを学習する。

- 加重平均モードの場合
```python
reward = - (weights["cost_min"] * (holding_cost + backlog_cost)) \
         + (weights["profit_max"] * profit) \
         - (weights["env_min"] * env_impact)
```
例えば `weights["env_min"] = 0.3` なら、環境負荷を30%の割合で考慮する形になる。

## トラブルシューティング

### 学習が進まない場合
- **メモリサイズの確認**: `agent.memory` が十分なサンプルを持っているか確認
- **バッチサイズの調整**: `config["batch_size"]` を小さくしてみる
- **学習率の調整**: `DQNAgent` クラス内の `lr` パラメータを調整
- **イプシロン減衰率の調整**: `epsilon_decay` を調整して探索と活用のバランスを変更

### NumPy/Matplotlibのエラーが出る場合
このエラーはNumPy 2.xとMatplotlibの互換性問題によるものです。以下の方法で解決できます：

1. セットアップスクリプトを使用して互換性のあるバージョンをインストール：
   ```bash
   ./setup.sh
   source venv/bin/activate
   ```

2. 手動でNumPyをダウングレードする：
   ```bash
   pip install numpy==1.26.4
   ```

3. 仮想環境を使用せずに直接ダウングレードする場合（非推奨）：
   ```bash
   pip install numpy==1.26.4 --force-reinstall
   ```
