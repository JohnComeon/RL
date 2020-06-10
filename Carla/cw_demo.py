

import carla
import random
import os
import sys
import glob
import time
import numpy as np
#from matplotlib import pyplot as plt
import cv2

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass



IM_WIDTH = 320
IM_HEIGHT = 240
actor_list = []

def process_img(image,I_images):
    i = np.array(image.raw_data)
    #print(i.shape)   # (307200,0)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))  # rgba
    i3 = i2[:,:,:3]
    #print(i3[1,1,:]) 
      #cv2.imshow("",i)
    #if cv2.waitKey(10) & 0xff==27:
    #    pass
    #plt.imshow(i[:,:,::-1]/255.0)
    #plt.imshow(1)
    I_images.append(i3)
    return #i/255.0



try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)
    
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter("model3")[0]
    print(bp)

    spwan_point = random.choice(world.get_map().get_spawn_points())
    #spwan_point = carla.Transform(carla.Location(179.470039, y=104.550026, z=1.370000), carla.Rotation(pitch=0.000000, yaw=-179.999634, roll=0.000000))
    print(spwan_point)
    vehicle = world.spawn_actor(bp, spwan_point)
    vehicle.set_autopilot(True)
    #vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    actor_list.append(vehicle)

    cam_bp = blueprint_library.find("sensor.camera.rgb")  
    cam_bp.set_attribute("image_size_x", str(IM_WIDTH))
    cam_bp.set_attribute("image_size_y", str(IM_HEIGHT))
    cam_bp.set_attribute("fov", "60")

    spwan_point2 = carla.Transform(carla.Location(x=2.5,z=0.7))
    sensor =  world.spawn_actor(cam_bp,spwan_point2, attach_to=vehicle)
    actor_list.append(sensor)
    #data = process_img(data)
    I_images = []
    sensor.listen(lambda data: process_img(data, I_images))
   #sensor.listen(data)

    time.sleep(10)
    
    for i in I_images:
	cv2.imshow('image', i)
	cv2.waitKey(10)


finally:
    for actor in actor_list:
        actor.destroy()
    print("ALL Cleaned up!")




