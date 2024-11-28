set steps 5000000
set strategy "default"

python playground.py ppo $steps $strategy
python playground.py ddpg $steps $strategy
python playground.py sac $steps $strategy
python playground.py td3 $steps $strategy
python playground.py trpo $steps $strategy


