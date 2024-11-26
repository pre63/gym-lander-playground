set steps 1000
set strategy "default"

python playground.py ppo $steps $strategy
python playground.py ddpg $steps $strategy
python playground.py sac $steps $strategy
python playground.py td3 $steps $strategy
python playground.py trpo $steps $strategy

set steps 1000000

python playground.py ppo $steps $strategy
python playground.py ddpg $steps $strategy
python playground.py sac $steps $strategy
python playground.py td3 $steps $strategy
python playground.py trpo $steps $strategy

set strategy "proximity"

python playground.py ppo $steps $strategy
python playground.py ddpg $steps $strategy
python playground.py sac $steps $strategy
python playground.py td3 $steps $strategy
python playground.py trpo $steps $strategy

set strategy "energy"

python playground.py ppo $steps $strategy
python playground.py ddpg $steps $strategy
python playground.py sac $steps $strategy
python playground.py td3 $steps $strategy
python playground.py trpo $steps $strategy

set strategy "combined"

python playground.py ppo $steps $strategy
python playground.py ddpg $steps $strategy
python playground.py sac $steps $strategy
python playground.py td3 $steps $strategy
python playground.py trpo $steps $strategy

set steps 1000000*5
set strategy "default"

python playground.py ppo $steps $strategy
python playground.py ddpg $steps $strategy
python playground.py sac $steps $strategy
python playground.py td3 $steps $strategy
python playground.py trpo $steps $strategy


