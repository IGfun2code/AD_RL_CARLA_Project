# AD_RL_CARLA_Project
### Activate the environment:
.\\.venv\Scripts\activate
### Bring up the simulation:
.\CarlaUE4.exe -quality-level=Low -carla-rpc-port=2000 -carla-streaming-port=2001 

### If connections are not working, then check what is listening in the port
netstat -ano | findstr :2000

### If there are multiple things listening, clear up the port:
taskkill /F /IM CarlaUE4.exe\
taskkill /F /IM CarlaUE4-Win64-Shipping.exe

### Waypoint and Map Information for both scenarios:
Left Turn :::: Ego Vehicle 224 - 83  :::: Traffic 173 - 227 :::: Map - Town 01\
Highway   :::: Ego Vehicle 54  - 211 :::: Traffic 203 - 137 :::: Map - Town 04

### Call to run the left_turn environment in town 01:
python .\scene_runner_baseline.py --town Town01_Opt --hide-layers --freeze-lights `
>>   --ego-spawn 224 --ego-dest 83 --oncoming-anchor 173 --oncoming-dest 227 `
>>   --traffic-stream --mean-headway 1.00 --burst-prob 0.5 --traffic-profile normal `
>>   --agent behavior --behavior aggressive --episodes 100 --spacing_m 10 --out results/left_turn/test_behavior_aggressive_100_normal.csv

### Call to run the left_turn environment in town 01:
python .\scene_runner_baseline.py --town Town04_Opt --hide-layers --freeze-lights `
>>   --ego-spawn 54 --ego-dest 211 --oncoming-anchor 203 --oncoming-dest 137 `
>>   --traffic-stream --mean-headway 1.00 --burst-prob 0.5 --traffic-profile aggressive `
>>   --agent behavior --behavior normal --episodes 100 --spacing_m 2.5 --out results/highway_merge/test_behavior_normal_100_aggressive.csv

### Get the data into a csv using the helper function:
python tools\analyze_results.py results\highway_merge\test_*.csv --out results/compare/highway_merge.csv
