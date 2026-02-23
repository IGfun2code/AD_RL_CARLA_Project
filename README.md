# AD_RL_CARLA_Project
### Activate the environment:
.\\.venv\Scripts\activate
### Bring up the simulation:
.\CarlaUE4.exe -quality-level=Low -carla-rpc-port=2000 -carla-streaming-port=2001 

### If connections are not working, then check what is listening in the port
netstat -ano | findstr :2000

### If there are multiple things listening, clear up the port:
taskkill /F /IM CarlaUE4.exe
taskkill /F /IM CarlaUE4-Win64-Shipping.exe



