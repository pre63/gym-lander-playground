#!/usr/bin/env fish

# Define parameters
set models ppo sac td3 totd trpo ddpg laber mac
set strategies default proximity energy_efficient combined
set episodes 10 100 200 500 1000 2000

# Loop over episodes, models, and strategies
for episodes in $episodes
    for strategy in $strategies
        for model in $models
            echo "Running: python playground.py $model $episodes $strategy"
            python playground.py $model $episodes $strategy &
        end
        wait
    end
end

echo "All experiments completed."
