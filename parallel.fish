#!/usr/bin/env fish

function run_with_limit
    while true
        set job_count (jobs | wc -l)
        if test $job_count -lt 3
            break
        end
        wait -n
    end
end

set strategies default proximity energy_efficient combined
set episodes_list 20 50 100 200 500 1000 2000

for episodes in $episodes_list
    for strategy in $strategies
        python suite.py $episodes $strategy &
        run_with_limit
    end
end

wait

echo "All experiments completed."
