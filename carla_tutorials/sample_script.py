import carla
import random

'''
Connect with the client and set the map
'''
# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
world = client.get_world()

# Set the map you want to load
client.load_world('Town10')

'''
Access the spectator location and such
'''
# Retrieve the spectator object
spectator = world.get_spectator()

# Get the location and rotation of the spectator through its transform
transform = spectator.get_transform()

location = transform.location
rotation = transform.rotation

# Set the spectator with an empty transform
spectator.set_transform(carla.Transform())
# This will set the spectator at the origin of the map, with 0 degrees
# pitch, yaw and roll - a good way to orient yourself in the map

'''
Adding NPCs
'''
# Get the blueprint library and filter for the vehicle blueprints
vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')

# Get the map's spawn points
spawn_points = world.get_map().get_spawn_points()

# Spawn 50 vehicles randomly distributed throughout the map 
# for each spawn point, we choose a random vehicle from the blueprint library
for i in range(0,50):
    world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))

'''
Add ego vehicle
'''
ego_bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')

ego_bp.set_attribute('role_name', 'hero')

ego_vehicle = world.spawn_actor(ego_bp, random.choice(spawn_points))

'''
Creating a sensor example (https://carla.readthedocs.io/en/latest/core_sensors/)
'''
# Create a transform to place the camera on top of the vehicle
camera_init_trans = carla.Transform(carla.Location(z=1.5))

# We create the camera through a blueprint that defines its properties
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

# We spawn the camera and attach it to our ego vehicle
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)

# Start camera with PyGame callback and save pngs to the disk folder called 'out'
camera.listen(lambda image: image.save_to_disk('out/%06d.png' % image.frame))

'''
Animate vehicles with traffic manager
'''
for vehicle in world.get_actors().filter('*vehicle*'):
    vehicle.set_autopilot(True)