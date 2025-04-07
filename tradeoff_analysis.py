import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import seaborn as sns
from beer_game_dqn import BeerGameEnv, DQNAgent, device

# === トレードオフ分析用の設定 ===
tradeoff_dir = f"output/tradeoff_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(tradeoff_dir, exist_ok=True)

# 分析対象のゴール設定
goals = [
    "cost_min",      # Cost Minimization
    "profit_max",    # Profit Maximization
    "env_min",       # Environmental Impact Minimization
    # Weighted patterns
    # "weighted_balanced",    # Balanced
    # "weighted_cost_profit", # Cost-Profit Focus
    # "weighted_cost_env",    # Cost-Environment Focus
    # "weighted_profit_env"   # Profit-Environment Focus
]

# 重み付けゴールの設定を定義
weight_settings = {
    "weighted_balanced": {"cost_min": 0.4, "profit_max": 0.3, "env_min": 0.3},
    "weighted_cost_profit": {"cost_min": 0.5, "profit_max": 0.5, "env_min": 0.0},
    "weighted_cost_env": {"cost_min": 0.5, "profit_max": 0.0, "env_min": 0.5},
    "weighted_profit_env": {"cost_min": 0.0, "profit_max": 0.5, "env_min": 0.5}
}

# 基本設定
base_config = {
    "time_unit": "day",
    "batch_size": 256,
    "num_episodes": 600,  # 分析用に少なめのエピソード数
    "learning_rate": 0.0005,
    "discount_factor": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.1,
    "epsilon_decay": 0.998
}

# 各ゴールでのトレーニング結果を保存
model_paths = {}
train_metrics = {}
test_results = {}

# === 各ゴールでの学習と評価 ===
def train_for_goal(goal, config):
    """指定したゴールに対してモデルを学習"""
    print(f"\n=== Starting Training: {goal} ===")
    
    # 環境とエージェントの初期化
    env = BeerGameEnv()
    agent = DQNAgent(state_size=4, action_size=10)
    
    reward_history = []
    loss_history = []
    avg_reward_history = []
    
    # トレーニングループ
    for episode in range(config["num_episodes"]):
        state = env.reset()
        total_reward = 0
        episode_loss = []
        
        for step in range(env.max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            
            if len(agent.memory) >= config["batch_size"]:
                loss = agent.train(config["batch_size"])
                episode_loss.append(loss)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # エピソード終了時の処理
        reward_history.append(total_reward)
        avg_reward = np.mean(reward_history[-100:]) if len(reward_history) >= 100 else np.mean(reward_history)
        avg_reward_history.append(avg_reward)
        loss_history.append(np.mean(episode_loss) if episode_loss else 0)
        
        # 10エピソードごとにターゲットネットワークを更新
        if episode % 10 == 0:
            agent.update_target_network()
            print(f"Episode {episode}/{config['num_episodes']}: Reward={total_reward:.2f}, Avg={avg_reward:.2f}")
    
    # モデルの保存
    model_path = os.path.join(tradeoff_dir, f"model_{goal}.pth")
    torch.save(agent.model.state_dict(), model_path)
    model_paths[goal] = model_path
    
    # 学習指標の保存
    train_metrics[goal] = {
        "rewards": reward_history,
        "avg_rewards": avg_reward_history,
        "losses": loss_history
    }
    
    print(f"Training Completed: {goal}")
    return agent, env

def evaluate_model(goal, agent, env, num_episodes=10):
    """学習したモデルの評価を行う"""
    print(f"\n=== Model Evaluation: {goal} ===")
    
    # 各評価指標の保存用
    metrics = {
        "total_reward": [],
        "avg_inventory": [],
        "avg_backlog": [],
        "avg_holding_cost": [],
        "avg_backlog_cost": [],
        "avg_profit": [],
        "avg_env_impact": []
    }
    
    for i in range(num_episodes):
        state = env.reset()
        episode_info = []
        
        for step in range(env.max_steps):
            # 評価時は完全にモデルに従う（ε=0）
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = torch.argmax(agent.model(state_tensor)).item()
            
            next_state, reward, done, info = env.step(action)
            episode_info.append(info)
            state = next_state
            
            if done:
                break
        
        # エピソードの合計報酬
        metrics["total_reward"].append(sum(env.reward_history))
        
        # その他の指標の平均値を計算
        metrics["avg_inventory"].append(np.mean([info["inventory"] for info in episode_info]))
        metrics["avg_backlog"].append(np.mean([info["backlog"] for info in episode_info]))
        metrics["avg_holding_cost"].append(np.mean([info["holding_cost"] for info in episode_info]))
        metrics["avg_backlog_cost"].append(np.mean([info["backlog_cost"] for info in episode_info]))
        metrics["avg_profit"].append(np.mean([info["profit"] for info in episode_info]))
        metrics["avg_env_impact"].append(np.mean([info["env_impact"] for info in episode_info]))
        
    # 全エピソードの平均を計算
    result = {metric: np.mean(values) for metric, values in metrics.items()}
    print(f"Evaluation Results: {goal}")
    print(f"Average Reward: {result['total_reward']:.2f}")
    print(f"Average Inventory: {result['avg_inventory']:.2f}")
    print(f"Average Backlog: {result['avg_backlog']:.2f}")
    
    return result

# === 各ゴールでのモデル学習と評価 ===
for goal in goals:
    # 設定の更新
    goal_config = base_config.copy()
    
    if goal.startswith("weighted_"):
        # 重み付けゴールの場合
        goal_config["goal"] = "weighted"
        goal_config["reward_weights"] = weight_settings[goal]
    else:
        # 単一ゴールの場合
        goal_config["goal"] = goal
        goal_config["reward_weights"] = {"cost_min": 1.0, "profit_max": 0.0, "env_min": 0.0}
    
    # モデルの学習
    agent, env = train_for_goal(goal, goal_config)
    
    # モデルの評価
    test_results[goal] = evaluate_model(goal, agent, env)

# === トレードオフ関係の可視化 ===

# 1. レーダーチャートで各ゴールの特性を比較
def create_radar_chart():
    # 比較する指標
    metrics = [
        "total_reward", 
        "avg_inventory", 
        "avg_backlog", 
        "avg_holding_cost", 
        "avg_backlog_cost", 
        "avg_profit", 
        "avg_env_impact"
    ]
    
    # データの正規化
    normalized_data = {}
    for metric in metrics:
        values = [test_results[goal][metric] for goal in goals]
        
        # 環境影響とコストは小さいほど良いため、反転させる
        if metric in ["avg_holding_cost", "avg_backlog_cost", "avg_env_impact", "avg_backlog"]:
            min_val = min(values)
            max_val = max(values)
            if max_val - min_val > 0:
                normalized_data[metric] = [1 - ((val - min_val) / (max_val - min_val)) for val in values]
            else:
                normalized_data[metric] = [0.5 for _ in values]
        else:
            # 大きいほど良い指標は通常の正規化
            min_val = min(values)
            max_val = max(values)
            if max_val - min_val > 0:
                normalized_data[metric] = [(val - min_val) / (max_val - min_val) for val in values]
            else:
                normalized_data[metric] = [0.5 for _ in values]
    
    # レーダーチャートのプロット
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # 各指標の角度
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 円を閉じるために最初の点を繰り返す
    
    # 各ゴールのプロット
    for i, goal in enumerate(goals):
        values = [normalized_data[metric][i] for metric in metrics]
        values += values[:1]  # 円を閉じるために最初の値を繰り返す
        
        ax.plot(angles, values, linewidth=2, label=goal)
        ax.fill(angles, values, alpha=0.1)
    
    # チャートの設定
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([metric.replace("avg_", "") for metric in metrics])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_title("Comparison of Metrics Across Goals", size=15)
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    
    plt.savefig(os.path.join(tradeoff_dir, "radar_chart.png"))
    plt.close()

# 2. ペアプロットで指標間の関係を可視化
def create_pair_plot():
    # 結果をデータフレームに変換
    df_data = []
    for goal in goals:
        row = test_results[goal].copy()
        row["goal"] = goal
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # ペアプロット
    metrics = ["avg_holding_cost", "avg_backlog_cost", "avg_profit", "avg_env_impact"]
    sns.pairplot(df, vars=metrics, hue="goal", height=2.5)
    plt.savefig(os.path.join(tradeoff_dir, "pair_plot.png"))
    plt.close()

# 3. 棒グラフで各目標の結果を比較
def create_bar_charts():
    # 主要な指標の棒グラフを作成
    metrics = [
        ("avg_holding_cost", "Average Holding Cost"),
        ("avg_backlog_cost", "Average Backlog Cost"),
        ("avg_profit", "Average Profit"),
        ("avg_env_impact", "Average Environmental Impact")
    ]
    
    plt.figure(figsize=(15, 12))
    
    for i, (metric, title) in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        values = [test_results[goal][metric] for goal in goals]
        
        plt.bar(goals, values)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 値のラベル追加
        for j, v in enumerate(values):
            plt.text(j, v + 0.02 * max(values), f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(tradeoff_dir, "comparison_bar_charts.png"))
    plt.close()

# 4. 学習曲線の比較
def plot_learning_curves():
    plt.figure(figsize=(15, 10))
    
    # 報酬の推移
    plt.subplot(2, 1, 1)
    for goal in goals:
        plt.plot(train_metrics[goal]["avg_rewards"], label=goal)
    
    plt.title("Average Reward Progress")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)
    
    # 損失の推移
    plt.subplot(2, 1, 2)
    for goal in goals:
        plt.plot(train_metrics[goal]["losses"], label=goal)
    
    plt.title("Loss Progress")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(tradeoff_dir, "learning_curves.png"))
    plt.close()

# 5. トレードオフの散布図マトリックス
def create_tradeoff_matrix():
    # コスト vs 利益 vs 環境のトレードオフを可視化
    fig = plt.figure(figsize=(12, 12))
    
    # データ準備
    costs = [test_results[goal]["avg_holding_cost"] + test_results[goal]["avg_backlog_cost"] for goal in goals]
    profits = [test_results[goal]["avg_profit"] for goal in goals]
    env_impacts = [test_results[goal]["avg_env_impact"] for goal in goals]
    
    # コスト vs 利益
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(costs, profits, c=range(len(goals)), cmap='viridis')
    for i, goal in enumerate(goals):
        ax1.annotate(goal, (costs[i], profits[i]))
    ax1.set_xlabel("Total Cost")
    ax1.set_ylabel("Profit")
    ax1.set_title("Cost vs Profit")
    ax1.grid(True)
    
    # コスト vs 環境影響
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(costs, env_impacts, c=range(len(goals)), cmap='viridis')
    for i, goal in enumerate(goals):
        ax2.annotate(goal, (costs[i], env_impacts[i]))
    ax2.set_xlabel("Total Cost")
    ax2.set_ylabel("Environmental Impact")
    ax2.set_title("Cost vs Environmental Impact")
    ax2.grid(True)
    
    # 利益 vs 環境影響
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(profits, env_impacts, c=range(len(goals)), cmap='viridis')
    for i, goal in enumerate(goals):
        ax3.annotate(goal, (profits[i], env_impacts[i]))
    ax3.set_xlabel("Profit")
    ax3.set_ylabel("Environmental Impact")
    ax3.set_title("Profit vs Environmental Impact")
    ax3.grid(True)
    
    # 3D プロット
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.scatter(costs, profits, env_impacts, c=range(len(goals)), cmap='viridis')
    for i, goal in enumerate(goals):
        ax4.text(costs[i], profits[i], env_impacts[i], goal)
    ax4.set_xlabel("Total Cost")
    ax4.set_ylabel("Profit")
    ax4.set_zlabel("Environmental Impact")
    ax4.set_title("Cost vs Profit vs Environmental Impact")
    
    plt.tight_layout()
    plt.savefig(os.path.join(tradeoff_dir, "tradeoff_matrix.png"))
    plt.close()

# 6. 結果のCSV出力
def export_results_to_csv():
    # 結果をデータフレームに変換
    df_data = []
    for goal in goals:
        row = test_results[goal].copy()
        
        # 重み付け目標の場合、その設定も記録
        if goal.startswith("weighted_"):
            for weight_type, value in weight_settings[goal].items():
                row[f"weight_{weight_type}"] = value
        else:
            for weight_type in ["cost_min", "profit_max", "env_min"]:
                row[f"weight_{weight_type}"] = 1.0 if goal == weight_type else 0.0
        
        row["goal"] = goal
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # CSVに保存
    df.to_csv(os.path.join(tradeoff_dir, "results.csv"), index=False)
    
    # 主要指標を抽出したテーブルを作成
    key_metrics = ["goal", "total_reward", "avg_inventory", "avg_backlog", 
                   "avg_holding_cost", "avg_backlog_cost", "avg_profit", "avg_env_impact"]
    
    # 重み情報の追加
    weight_cols = [col for col in df.columns if col.startswith("weight_")]
    key_metrics.extend(weight_cols)
    
    df_key = df[key_metrics]
    df_key.to_csv(os.path.join(tradeoff_dir, "key_metrics.csv"), index=False)
    
    # テキスト形式でも保存（読みやすさのため）
    with open(os.path.join(tradeoff_dir, "summary.txt"), "w") as f:
        f.write("=== Trade-off Analysis Results ===\n\n")
        
        for goal in goals:
            f.write(f"\n{goal}:\n")
            for metric, value in test_results[goal].items():
                f.write(f"  {metric}: {value:.4f}\n")
            
            if goal.startswith("weighted_"):
                f.write("  Weight Settings:\n")
                for weight_type, value in weight_settings[goal].items():
                    f.write(f"    {weight_type}: {value:.2f}\n")
            f.write("\n")

# 全ての可視化を実行
create_radar_chart()
create_pair_plot()
create_bar_charts()
plot_learning_curves()
create_tradeoff_matrix()
export_results_to_csv()

print(f"\nAnalysis results have been saved to: {tradeoff_dir}")