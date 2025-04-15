import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
import datetime

from replay_buffer import ReplayBuffer
from agents import DQNAgent, MCTSDQNAgent, MiniMaxAgent
from utils import (
    get_connect_four_game, 
    play_game, 
    evaluate_agent_elo,
    set_seed,
    visualize_board, 
    ensure_dir,
    calculate_moving_average
)

def train(config):
    """Train an agent
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Trained agent
    """
    # Configuration parameters
    num_episodes = config["num_episodes"]
    batch_size = config["batch_size"]
    replay_buffer_size = config["replay_buffer_size"]
    dqn_learning_rate = config["dqn_learning_rate"]
    gamma = config["gamma"]
    target_update_freq = config["target_update_freq"]
    epsilon_start = config["epsilon_start"]
    epsilon_end = config["epsilon_end"]
    epsilon_decay = config["epsilon_decay"]
    training_freq = config["training_freq"]
    lambda_mix = config["lambda_mix"]
    eval_freq = config["eval_freq"]
    eval_minimax_depth = config.get("eval_minimax_depth", 7)
    checkpoint_freq = config["checkpoint_freq"]
    checkpoint_dir = config["checkpoint_dir"]
    device = config["device"]
    seed = config["seed"]
    
    # Agent type and related parameters
    agent_type = config.get("agent_type", "dqn")
    mcts_simulations = config.get("mcts_simulations")
    use_dqn_for_mcts = config.get("use_dqn_for_mcts", False)
    
    # Ensure checkpoint directory exists
    ensure_dir(checkpoint_dir)
    
    set_seed(seed)
    
    # Create game environment
    game = get_connect_four_game()
    
    # Create agent
    input_shape = (3, 6, 7)  # Connect4 observation space shape
    action_size = 7  # Connect4 has 7 columns to place tokens
    
    if agent_type == "dqn":
        # Create standard DQN agent
        agent = DQNAgent(
            input_shape=input_shape,
            action_size=action_size,
            device=device,
            learning_rate=dqn_learning_rate,
            gamma=gamma
        )
    elif agent_type == "mcts_dqn":
        # Create MCTS-DQN hybrid agent
        agent = MCTSDQNAgent(
            input_shape=input_shape,
            action_size=action_size,
            device=device,
            learning_rate=dqn_learning_rate,
            gamma=gamma,
            num_simulations=mcts_simulations,
            uct_c=config.get("uct_c", 2.0),
            max_nodes=config.get("max_nodes", 10000),
            solve=config.get("solve", True),
            use_dqn_evaluator=use_dqn_for_mcts,
            n_rollouts=1
        )
    elif agent_type == "minimax":
        # Create MiniMax agent
        agent = MiniMaxAgent(
            input_shape=input_shape,
            action_size=action_size,
            device=device,
            max_depth=config.get("minimax_depth", 4)
        )
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    
    # If MiniMax agent, no training needed
    if agent_type == "minimax":
        # Save and return the agent
        final_model_path = os.path.join(checkpoint_dir, "minimax_model.pth")
        print(f"MiniMax agent created with depth {agent.max_depth}.")
        return agent
    
    # Create replay buffer for other agent types
    replay_buffer = ReplayBuffer(capacity=replay_buffer_size)
    
    # Record training metrics
    rewards_0 = []
    rewards_1 = []
    losses = []
    epsilon_values = []
    game_length = []
    
    # Initialize Elo ratings and history
    initial_elo = 1200
    current_elos = {
        'agent_p1_elo': initial_elo, 'agent_p2_elo': initial_elo,
        'minimax_p1_elo': initial_elo, 'minimax_p2_elo': initial_elo
    }
    elo_history = {
        'episodes': [],
        'agent_p1_elo': [], 'agent_p2_elo': [],
        'minimax_p1_elo': [], 'minimax_p2_elo': []
    }
    
    # Create the fixed Minimax opponent for evaluation
    eval_opponent = MiniMaxAgent(
        input_shape=input_shape,
        action_size=action_size,
        device='cpu',
        max_depth=eval_minimax_depth
    )
    print(f"Evaluation opponent: MiniMaxAgent with depth {eval_minimax_depth}")
    
    # Training main loop
    start_time = time.time()
    for episode in tqdm(range(1, num_episodes + 1), desc="Training Progress", unit="episodes"):
        # Calculate current epsilon value (exploration rate)
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (episode / num_episodes))
        epsilon_values.append(epsilon)
        
        # Self-play and collect experiences
        returns, final_state, length = play_game(
            agent1=agent,
            agent2=agent,
            epsilon=epsilon,
            verbose=False,
            collect_experience=True,
            replay_buffer=replay_buffer,
            lambda_mix=lambda_mix
        )
        
        # rewards0.append(returns[0])  # Record player 1's reward
        # rewards_1.append(returns[1])
        game_length.append(length)
        
        # Train the agent
        if len(replay_buffer) >= batch_size:
            for _ in range(training_freq):
                # Sample from replay buffer
                states, actions, rewards_batch, next_states, q_mcts_values, dones = replay_buffer.sample(batch_size)
                
                # Update DQN
                loss = agent.learn(states, actions, rewards_batch, next_states, q_mcts_values, dones, lambda_mix)
                losses.append(loss)
        
        # Periodically update target network
        if episode % target_update_freq == 0:
            if hasattr(agent, 'update_target_network'):
                agent.update_target_network()
        
        # Periodically evaluate agent performance using Elo
        if episode % eval_freq == 0 or episode == num_episodes:
            print(f"\n--- Evaluating Episode {episode}/{num_episodes} ---")
            # Evaluate current agent's Elo against fixed Minimax opponent
            updated_elos = evaluate_agent_elo(
                agent=agent,
                opponent_agent=eval_opponent,
                current_elos=current_elos,
                k_factor=config.get("elo_k_factor", 32)
            )
            current_elos = updated_elos
            
            # Store Elo history
            elo_history['episodes'].append(episode)
            elo_history['agent_p1_elo'].append(current_elos['agent_p1_elo'])
            elo_history['agent_p2_elo'].append(current_elos['agent_p2_elo'])
            elo_history['minimax_p1_elo'].append(current_elos['minimax_p1_elo'])
            elo_history['minimax_p2_elo'].append(current_elos['minimax_p2_elo'])
            
            # Print evaluation results
            steps_since_last_eval = training_freq * eval_freq
            avg_loss = np.mean(losses[-steps_since_last_eval:]) if losses else 0
            print(f"Episode {episode}/{num_episodes} | Epsilon: {epsilon:.4f} | Avg Loss: {avg_loss:.6f}")
            print(f"  Elo Scores: Agent(P1): {current_elos['agent_p1_elo']:.1f}, Agent(P2): {current_elos['agent_p2_elo']:.1f}")
            print(f"              Minimax(P1): {current_elos['minimax_p1_elo']:.1f}, Minimax(P2): {current_elos['minimax_p2_elo']:.1f}")
            print("-----------------------------------------")
        
        # Save checkpoints
        if episode % checkpoint_freq == 0 or episode == num_episodes:
            t = datetime.datetime.now().strftime('%d-%H_%m')
            checkpoint_path = os.path.join(checkpoint_dir, f"{t}_model_episode_{episode}.pth")
            if hasattr(agent, 'save'):
                agent.save(checkpoint_path)
    
    # Save final model
    t = datetime.datetime.now().strftime('%d-%H_%m')
    final_model_path = os.path.join(checkpoint_dir, f"{t}_final_{agent_type}_model.pth")
    if hasattr(agent, 'save'):
        agent.save(final_model_path)
        print(f"Training completed, final model saved at: {final_model_path}")
    else:
        print(f"Training completed for agent type {agent_type} (no save method applicable).")
    
    # Calculate training time
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")
    
    # Plot training curves
    if losses or elo_history['episodes']:
        num_plots = (1 if elo_history['episodes'] else 0) * 2 + (1 if losses else 0) + (1 if epsilon_values else 0) + (1 if game_length else 0)
        if num_plots == 0:
            print("No data to plot.")
            return agent
        
        plt.figure(figsize=(12, 4 * num_plots))
        plot_index = 1
        
        # Plot loss curve
        if losses:
            plt.subplot(num_plots, 1, plot_index)
            loss_ma_window = max(100, len(losses) // 50)
            loss_ma = calculate_moving_average(losses, window=loss_ma_window)
            if len(loss_ma) > 0:
                loss_steps = np.linspace(0, len(losses), len(loss_ma))
                plt.plot(loss_steps, loss_ma)
                plt.title(f'Training Loss (Moving Average, window={loss_ma_window})')
                plt.xlabel('Training Steps')
                plt.ylabel('Loss')
            else:
                plt.plot(losses)
                plt.title('Training Loss (Raw)')
                plt.xlabel('Training Steps')
                plt.ylabel('Loss')
            plot_index += 1
        
        # Plot epsilon curve
        if epsilon_values:
            plt.subplot(num_plots, 1, plot_index)
            plt.plot(range(1, num_episodes + 1), epsilon_values)
            plt.title('Exploration Rate (Epsilon)')
            plt.xlabel('Episodes')
            plt.ylabel('Epsilon')
            plot_index += 1
        
        # Plot game length
        if game_length:
            plt.subplot(num_plots, 1, plot_index)
            length_ma = calculate_moving_average(game_length, window=5)
            length_steps = np.linspace(0, len(game_length), len(length_ma))
            plt.plot(length_steps, length_ma)
            plt.title('Game Length (Actions, moving average window=5)')
            plt.xlabel('Episodes')
            plt.ylabel('Length')
            plot_index += 1
        
        # Plot Elo P1 curve
        if elo_history['episodes']:
            plt.subplot(num_plots, 1, plot_index)
            plt.plot(elo_history['episodes'], elo_history['agent_p1_elo'], label=f'Agent Elo (P1)', marker='.')
            plt.plot(elo_history['episodes'], elo_history['minimax_p1_elo'], label=f'Minimax Depth {eval_minimax_depth} Elo (P1)', linestyle='--', marker='.')
            plt.title('Elo Rating (Player 1 vs Minimax)')
            plt.xlabel('Episodes')
            plt.ylabel('Elo')
            plt.legend()
            plt.grid(True)
            plot_index += 1
        
        # Plot Elo P2 curve
        if elo_history['episodes']:
            plt.subplot(num_plots, 1, plot_index)
            plt.plot(elo_history['episodes'], elo_history['agent_p2_elo'], label=f'Agent Elo (P2)', marker='.')
            plt.plot(elo_history['episodes'], elo_history['minimax_p2_elo'], label=f'Minimax Depth {eval_minimax_depth} Elo (P2)', linestyle='--', marker='.')
            plt.title('Elo Rating (Player 2 vs Minimax)')
            plt.xlabel('Episodes')
            plt.ylabel('Elo')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout(pad=2.0)
        plot_filename = f'training_metrics_{agent_type}_elo.png'
        plot_filepath = os.path.join(checkpoint_dir, plot_filename)
        plt.savefig(plot_filepath)
        print(f"Training plots saved to {plot_filepath}")
    
    return agent

if __name__ == "__main__":
    # Training configuration
    config = {
        "num_episodes": 5000,           # Number of training episodes
        "batch_size": 64,               # Training batch size
        "replay_buffer_size": 100000,    # Replay buffer capacity
        "dqn_learning_rate": 0.0001,    # DQN learning rate
        "gamma": 0.99,                  # Reward discount factor
        "target_update_freq": 500,      # Target network update frequency
        "epsilon_start": 1.0,           # Initial exploration rate
        "epsilon_end": 0.05,            # Final exploration rate
        "epsilon_decay": (1.0 - 0.05) / 5000, # Linear decay factor (recalculated for clarity, value depends on num_episodes)
        "training_freq": 4,             # Training steps per episode's experience
        "lambda_mix": 0.5,              # Mixing coefficient for MCTS and DQN targets (if applicable)
        "eval_freq": 1000,               # Evaluation frequency (in episodes)
        "eval_minimax_depth": 6,        # Depth of Minimax opponent for evaluation
        "elo_k_factor": 32,             # Elo K-factor for rating updates
        "checkpoint_freq": 1000,         # Checkpoint saving frequency
        "checkpoint_dir": "checkpoints",# Checkpoint directory
        "device": "cuda" if torch.cuda.is_available() else "cpu",  # Computing device
        
        # MCTS parameters
        "mcts_simulations": 500,         # MCTS simulations per step during self-play/action selection
        "uct_c": 2.0,                   # UCT exploration constant
        "max_nodes": 10000,             # Maximum nodes in MCTS search tree
        "dirichlet_alpha": 0.3,         # Dirichlet noise parameter (consider adding to MCTS agent if used)
        "dirichlet_noise": False,       # Whether to add Dirichlet noise (consider adding to MCTS agent if used)
        "solve": True,                  # Whether to solve terminal states in MCTS
        "use_dqn_for_mcts": True,      # Whether to use DQN to evaluate MCTS leaf nodes
        
        # Agent type
        "agent_type": "dqn",            # Options: "dqn", "mcts_dqn" (Minimax is not trained here)
        
        # MiniMax parameters
        "minimax_depth": 4,             # MiniMax search depth if creating a minimax agent directly
    }
    
    # Start training
    trained_agent = train(config)

    # Example of how to potentially use the trained agent after training
    if trained_agent and config['agent_type'] != 'minimax':
        print("\nTraining finished. Example: Playing one game against Minimax depth 4.")
        from utils import play_interactive_game, play_game # Ensure play_game is also imported if used here
        # play_interactive_game(trained_agent) # Play against human
        # Create an opponent for post-training test
        eval_opponent_final = MiniMaxAgent(input_shape=(3, 6, 7), action_size=7, device='cpu', max_depth=4)
        final_returns, _, _ = play_game(trained_agent, eval_opponent_final, verbose=True, collect_experience=False) # Use the modified play_game
        print(f"Final game vs Minimax Depth 4: Agent score = {final_returns[0]}, Minimax score = {final_returns[1]}")
        # You might want to play another game with players swapped
        final_returns_swapped, _, _ = play_game(eval_opponent_final, trained_agent, verbose=True, collect_experience=False)
        print(f"Final game (swapped) vs Minimax Depth 4: Minimax score = {final_returns_swapped[0]}, Agent score = {final_returns_swapped[1]}")