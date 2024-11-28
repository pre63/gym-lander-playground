set steps 5000000

python suite.py $steps "energy" &
python suite.py $steps "proximity" &
python suite.py $steps "default" &
python suite.py $steps "combined" 

wait