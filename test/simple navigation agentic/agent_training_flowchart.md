# Agent Training System Flowchart

```mermaid
flowchart TD
    Start([Start]) --> TrainOrEval{Training or Evaluation?}
    
    %% Training Path
    TrainOrEval -->|Training| T1[Create SimpleNavigationEnv]
    T1 --> T2[Initialize Environment State<br/>- Target: 5.0, 5.0<br/>- World Bounds: ±10<br/>- Max Steps: 100]
    T2 --> T3[Instantiate PPO Agent<br/>- MlpPolicy<br/>- Hyperparameters:<br/>  γ=0.99, n_steps=2048<br/>  batch_size=64, epochs=10<br/>  lr=3e-4]
    T3 --> T4[Train Agent<br/>total_timesteps=100,000]
    
    T4 --> TrainingLoop{Training Loop}
    TrainingLoop --> T5[Reset Environment<br/>agent_position = 0,0<br/>current_step = 0]
    T5 --> T6[Get Action from Policy<br/>action = dx, dy]
    T6 --> T7[Execute Step<br/>- Update position<br/>- Clip to bounds<br/>- Calculate reward]
    T7 --> T8[Compute Reward<br/>reward = exp-0.5 × distance²]
    T8 --> T9{Reached<br/>Goal?}
    
    T9 -->|Yes distance < 0.1| T10[Add Bonus Reward +10<br/>terminated = True]
    T9 -->|No| T11{Max Steps<br/>Reached?}
    T11 -->|Yes| T12[truncated = True]
    T11 -->|No| T13[Continue Episode]
    
    T10 --> T14[Collect Experience]
    T12 --> T14
    T13 --> T6
    T14 --> T15{Training<br/>Complete?}
    T15 -->|No| TrainingLoop
    T15 -->|Yes| T16[Save Model<br/>ppo_simple_navigation.zip]
    T16 --> T17[Close Environment]
    T17 --> End([End])
    
    %% Evaluation Path
    TrainOrEval -->|Evaluation| E1[Create SimpleNavigationEnv]
    E1 --> E2[Load Trained Model<br/>ppo_simple_navigation.zip]
    E2 --> E3{Model<br/>Found?}
    E3 -->|No| E4[Print Error<br/>Run train.py first]
    E4 --> End
    E3 -->|Yes| E5[Set num_episodes = 5]
    E5 --> EvalLoop{For Each<br/>Episode}
    
    EvalLoop --> E6[Reset Environment<br/>obs = 0,0]
    E6 --> E7[Initialize:<br/>terminated = False<br/>truncated = False<br/>total_reward = 0]
    E7 --> E8{Episode<br/>Done?}
    E8 -->|No| E9[Predict Action<br/>deterministic=True]
    E9 --> E10[Execute Step<br/>obs, reward, terminated,<br/>truncated, info = env.step]
    E10 --> E11[Accumulate Reward<br/>total_reward += reward]
    E11 --> E12[Render Visualization<br/>- Plot agent path<br/>- Show position<br/>- Display target]
    E12 --> E8
    E8 -->|Yes| E13[Print Episode Results<br/>Total Reward]
    E13 --> E14{More<br/>Episodes?}
    E14 -->|Yes| EvalLoop
    E14 -->|No| E15[Close Environment]
    E15 --> End
    
    %% Environment Details Box
    subgraph EnvDetails[Environment Components]
        direction TB
        ED1[Observation Space<br/>Box: -10 to +10, shape=2<br/>Agent x,y position]
        ED2[Action Space<br/>Box: -1 to +1, shape=2<br/>dx, dy movement]
        ED3[Reward Function<br/>exp-0.5 × distance² + goal bonus]
    end
    
    %% PPO Update Process
    subgraph PPOProcess[PPO Update Mechanism]
        direction TB
        P1[Collect n_steps experiences]
        P1 --> P2[Split into mini-batches<br/>batch_size=64]
        P2 --> P3[Update policy n_epochs times<br/>using clipped objective]
        P3 --> P4[Apply learning_rate=3e-4]
    end
    
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style TrainOrEval fill:#FFD700
    style TrainingLoop fill:#87CEEB
    style EvalLoop fill:#87CEEB
    style EnvDetails fill:#F0F8FF
    style PPOProcess fill:#FFF0E6
```

## Key Components

### 1. **Training Pipeline** (`train.py`)
- Creates environment with 2D navigation task
- Initializes PPO agent with MLP policy
- Trains for 100,000 timesteps using experience collection and policy updates
- Saves trained model

### 2. **Environment** (`simple_navigation_env.py`)
- **State Space**: Agent's (x, y) position in [-10, 10]²
- **Action Space**: Movement vector (dx, dy) in [-1, 1]²
- **Reward**: Exponential decay based on distance to target (5, 5)
- **Episode End**: Goal reached (distance < 0.1) or max steps (100)

### 3. **Evaluation Pipeline** (`evaluate.py`)
- Loads saved model
- Runs 5 evaluation episodes with deterministic policy
- Visualizes agent's trajectory in real-time
- Reports total reward per episode

### 4. **PPO Algorithm**
- **Hyperparameters**: γ=0.99, n_steps=2048, batch_size=64, epochs=10, lr=3e-4
- Collects experiences, updates policy using clipped objective
- Logs training progress to TensorBoard
```
