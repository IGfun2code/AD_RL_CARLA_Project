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

### Call to run the left_turn environment in town 01:
python .\scene_runner_baseline.py --town Town01_Opt --hide-layers --freeze-lights `
>>   --ego-spawn 224 --ego-dest 83 --oncoming-anchor 173 --oncoming-dest 227 `
>>   --traffic-stream --mean-headway 1.8 --burst-prob 0.35 --traffic-profile aggressive `
>>   --agent behavior --behavior aggressive --episodes 100 --out results/left_turn/behavior_aggressive_100_aggressive.csv


Left Turn :::: Ego Vehicle 224 - 83  :::: Traffic 173 - 227 :::: Map - Town 01\
Highway   :::: Ego Vehicle 54  - 211 :::: Traffic 203 - 137 :::: Map - Town 04

