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

## 環境構築
### 🔹　リポジトリのクローン
```bash
git clone https://github.com/TaichiEto/beer_game
```
### 🔹 必要なライブラリ
以下のライブラリをインストールしてください。
```bash
pip install numpy matplotlib torch
```

### 🔹 実行方法
以下のコマンドで学習を開始できます。
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
}
```

- **`time_unit`**: シミュレーションの時間単位を設定（週・日・月）
- **`goal`**: 学習の目的（コスト最小化、利益最大化、環境負荷削減、または加重平均）
- **`reward_weights`**: 加重平均モードでの目的別の重みを設定

## 出力
1. **学習の進捗（エピソードごとの報酬）**
   ```
   Episode 0: Total Reward: -1200.50
   Episode 10: Total Reward: -980.30
   ...
   ```
2. **学習曲線の可視化**
   - `matplotlib` を使用して学習の進行状況をプロット
3. **テスト後の平均報酬**
   ```
   Test Average Reward: -850.20
   ```

## 人間プレイヤーとの比較機能
### 🔹 追加機能
🔹 **今後実装予定: GUI または CLI で手動操作可能**  
🔹 **今後実装予定: AI と人間のプレイ結果（在庫変動・コスト・報酬）を比較**  
🔹 **今後実装予定: AI vs 人間の評価指標（コスト最小化、利益最大化、安定性 など）を導入**  

## 今後の改良点
- **報酬関数の最適化**
- **強化学習アルゴリズムの追加（PPO、DDPGなど）**
- **プロジェクトマネジメントモデルへの拡張**

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
