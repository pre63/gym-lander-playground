#!/usr/bin/env fish

set strategies default proximity energy_efficient combined
set episodes 10 20

# Loop over episodes, models, and strategies
for strategy in $strategies
  for episodes in $episodes
    python suite.py $episodes $strategy &
  end
end
wait

set episodes  50 100

# Loop over episodes, models, and strategies
for strategy in $strategies
  for episodes in $episodes
    python suite.py $episodes $strategy &
  end
end
wait

set episodes 200 500

# Loop over episodes, models, and strategies
for strategy in $strategies
  for episodes in $episodes
    python suite.py $episodes $strategy &
  end
end
wait

set episodes 1000 2000

# Loop over episodes, models, and strategies
for strategy in $strategies
  for episodes in $episodes
    python suite.py $episodes $strategy &
  end
end
wait

echo "All experiments completed."
