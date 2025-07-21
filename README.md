# Final_REL_Reward_Hacking
This resporitory is inspired form the paper "[Correlated Proxies: A New Definition and Improved Mitigation for Reward Hacking](https://arxiv.org/abs/2403.03185)".

When I try to imitate the Tomato-watering environment but failed at the end. But then I use it as a reference and build an environment called ```Simplified_tomato_env```

### 🧩 Environment Constants

#### 🟫 Tile Types (States)
| Constant  | Value | Description     |
|-----------|--------|-----------------|
| `EMPTY`   | 0      | Empty tile      |
| `AGENT`   | 1      | The robot agent |
| `BUCKET`  | 2      | Water bucket    |
| `WATERED` | 3      | Watered tomato  |
| `DRY`     | 4      | Dry tomato      |
| `WALL`    | 5      | Wall / obstacle |

#### 🎮 Actions
| Constant | Value | Description      |
|----------|--------|------------------|
| `RIGHT`  | 0      | Move right       |
| `LEFT`   | 1      | Move left        |
| `DOWN`   | 2      | Move down        |
| `UP`     | 3      | Move up          |
| `NOOP`   | 4      | Do nothing       |

It will seperate into two cases when the agent try to bypass the rule by not watering the tomato to get true reward, instead it will try to get to the sprinkler cell( in this code is the bucket) to get the high proxy reward, in the paper the sprinkler state where the agent perceives all tomatoes as being watered and thus receives high proxy reward but no true reward.

So there would be two cases: one is the agent 'try-hard' to water all the tomato in the environmennt, sencond is the agent only want to get high proxy reward by getting to the bucket state but ignore to water the tomato.

---

## 🖼️ Visual Comparison

### The original tomato_environment from the paper
<p align="center">
  <img src="images/from_paper.png" width="300"/>
</p>

### The environment I recreated
<p align="center">
  <img src="images/self_built.png" width="300"/>
</p>

---
## Installation

Firstly to run the project you need to import all the libraries by running this line

    pip install -r requirements.txt

to install dependencies.

Next, you only need to run `simplyfied_tomato_env.py` file in the `env` folder in order to create the environment


##Training Process
By running the `train_ppo.py` file in the `training` folder you can train the agent with PPO
Next, you can use the `compare.py` file in the same folder to 
