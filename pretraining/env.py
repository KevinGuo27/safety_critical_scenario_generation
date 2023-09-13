import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm
from tensordict.nn import TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase
from torch import nn

from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp
import random

# create an enum class of vehicle, pedestrian, cyclist


class ObjectTypes(Enum):
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3


class CarlaEnv(EnvBase):
    def __init__(discriminator, encoder):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.world.unload_map_layer(carla.MapLayer.All)
        self._make_specs()
        self.max_timestep = 91
        self.discriminator = discriminator
        self.encoder = encoder
        seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _reset(self, info):
        self.collision_hist = []
        self.info = info
        self.valid_list = None
        self.actor_list = []
        self.vehicle_list = []
        self.object_types = info['object_type']
        self.init_timestep = 0
        self.object_num = len(self.object_types)
        self.ids = []
        self.sdc = None
        for i in range(len(self.object_types)):
            self.ids.append(i)
            init_valid = info['valid'][self.init_timestep][i, 0]
            while init_valid != 1:
                self.init_timestep += 1
                init_valid = info['valid'][self.init_timestep][i, 0]
            object_type = self.object_types[i]

            init_x, init_y, init_z, init_heading_angle, init_velocity_x, init_velocity_y,
            init_vel_yaw, init_length, init_width, init_height, init_timestamp_micros,
            init_is_self_driving, init_valid = self.load_state(
                self.init_timestep, i)
            self.valid_list.append(init_valid)
            if init_valid == 1:
                self.create_actor(object_type, i, init_x, init_y, init_z, init_heading_angle,
                                  init_velocity_x, init_velocity_y, init_vel_yaw, init_length, init_width,
                                  init_height, init_timestamp_micros, init_is_self_driving)
        colsensor = self.blueprint_library.find("sensor.other.collision")
        col_transform = self.sdc.get_transform()
        self.colsensor = self.world.spawn_actor(
            colsensor, col_transform, attach_to=self.sdc)
        self.colsensor.listen(lambda event: self.collision_data(event))
        self.world.tick()
        x, y, z, heading_angle, velocity_x, velocity_y, vel_yaw, length, width, height, timestamp_micros, is_self_driving = self.load_all_object_states(
            self.init_timestep)
        out = TensorDict(
            {
                "x": torch.tensor(x, dtype=torch.float32),
                "y": torch.tensor(y, dtype=torch.float32),
                "z": torch.tensor(z, dtype=torch.float32),
                "heading_angle": torch.tensor(heading_angle, dtype=torch.float32),
                "velocity_x": torch.tensor(velocity_x, dtype=torch.float32),
                "velocity_y": torch.tensor(velocity_y, dtype=torch.float32),
                "vel_yaw": torch.tensor(vel_yaw, dtype=torch.float32),
                "time_step": torch.tensor(self.init_timestep, dtype=torch.float32),
            },
            []
        )
        return out

    def collision_data(self, event):
        self.collision_hist.append(event)

    def create_actor(self, actor_type, actor_id, x, y, z, heading_angle, velocity_x, velocity_y, vel_yaw, length, width, height, timestamp_micros, is_self_driving):
        if actor_type == ObjectTypes.VEHICLE:
            blueprint = self.find_match_vehicles(length, width, height)
            transform = carla.Transform(carla.Location(x=x, y=y, z=z),
                                        carla.Rotation(yaw=self.from_radian_to_degree(heading_angle)))
            vehicle = self.world.spawn_actor(blueprint, transform)
            self.actor_list.append((vehicle, actor_id))
            self.vehicle_list.append(vehicle)
            vehicle.set_velocity(carla.Vector3D(
                x=velocity_x, y=velocity_y, z=0))
            vehicle.set_angular_velocity(
                carla.Vector3D(x=0, y=0, z=vel_yaw))
            vehicle.set_simulate_physics(True)
            if init_is_self_driving == 1:
                self.sdc = vehicle
                self.sdc_id = i
        elif actor_type == ObjectTypes.PEDESTRIAN:
            blueprint = random.choice(
                self.blueprint_library.filter('walker.*'))
            transform = carla.Transform(carla.Location(x=x, y=y, z=z),
                                        carla.Rotation(yaw=self.from_radian_to_degree(heading_angle)))
            walker = self.world.spawn_actor(blueprint, transform)
            self.actor_list.append((walker, actor_id))
            walker.set_velocity(carla.Vector3D(
                x=velocity_x, y=velocity_y, z=0))
            walker.set_angular_velocity(
                carla.Vector3D(x=0, y=0, z=vel_yaw))
            walker.set_simulate_physics(True)
        elif actor_type == ObjectTypes.CYCLIST:
            blueprint = random.choice(
                self.blueprint_library.filter('vehicle.bicycle'))
            transform = carla.Transform(carla.Location(x=x, y=y, z=z),
                                        carla.Rotation(yaw=self.from_radian_to_degree(heading_angle)))
            cyclist = self.world.spawn_actor(blueprint, transform)
            self.actor_list.append((cyclist, actor_id))
            # self.actor_list is a list of tuples, each tuple contains the actor and the index of the actor in the info dict
            cyclist.set_velocity(carla.Vector3D(
                x=velocity_x, y=velocity_y, z=0))
            cyclist.set_angular_velocity(
                carla.Vector3D(x=0, y=0, z=vel_yaw))
            cyclist.set_simulate_physics(True)

    def from_radian_to_degree(self, radian):
        return radian*180/np.pi

    def load_all_object_states(self, timestep):
        x = info['x'][timestep]
        # the shape of x is (object_num, 1)
        y = info['y'][timestep]
        z = info['z'][timestep]
        heading_angle = info['heading_angle'][timestep]
        velocity_x = info['velocity_x'][timestep]
        velocity_y = info['velocity_y'][timestep]
        vel_yaw = info['vel_yaw'][timestep]
        length = info['length'][timestep]
        width = info['width'][timestep]
        height = info['height'][timestep]
        timestamp_micros = info['timestamp_micros'][timestep]
        is_self_driving = info['is_self_driving'][timestep]
        valid = info['valid'][timestep]
        return x, y, z, heading_angle, velocity_x, velocity_y, vel_yaw, length, width, height, timestamp_micros, is_self_driving

    def load_state(self, timestep, object_num):
        x = info['x'][timestep][object_num, 0]
        y = info['y'][timestep][object_num, 0]
        z = info['z'][timestep][object_num, 0]
        heading_angle = info['heading_angle'][timestep][object_num, 0]
        velocity_x = info['velocity_x'][timestep][object_num, 0]
        velocity_y = info['velocity_y'][timestep][object_num, 0]
        vel_yaw = info['vel_yaw'][timestep][object_num, 0]
        length = info['length'][timestep][object_num, 0]
        width = info['width'][timestep][object_num, 0]
        height = info['height'][timestep][object_num, 0]
        timestamp_micros = info['timestamp_micros'][timestep][object_num, 0]
        is_self_driving = info['is_self_driving'][timestep][object_num, 0]
        valid = info['valid'][timestep][object_num, 0]
        return x, y, z, heading_angle, velocity_x, velocity_y, vel_yaw, length, width, height, timestamp_micros, is_self_driving, valid

    def _step(self, tensordict):
        # action is a tensordict of [throttle, steer, brake]
        throttle = tensordict["throttle"]
        steer = tensordict["steer"]
        brake = tensordict["brake"]
        timestep = tensordict["timestep"]
        self.sdc.apply_control(carla.VehicleControl(throttle=throttle,
                                                    steer=steer, brake=brake))
        self.world.tick()
        new_x, new_y, new_z = self.sdc.get_location()
        new_heading_angle = self.sdc.get_transform().rotation.yaw
        new_velocity_x, new_velocity_y, _ = self.sdc.get_velocity()
        new_vel_yaw = self.sdc.get_angular_velocity().z
        self.init_timestep += 1
        x, y, z, heading_angle, velocity_x, velocity_y, vel_yaw, length, width, height, timestamp_micros, is_self_driving = self.load_all_object_states(
            self.init_timestep)
        x[self.sdc_id, 0] = new_x
        y[self.sdc_id, 0] = new_y
        z[self.sdc_id, 0] = new_z
        heading_angle[self.sdc_id, 0] = new_heading_angle
        velocity_x[self.sdc_id, 0] = new_velocity_x
        velocity_y[self.sdc_id, 0] = new_velocity_y
        vel_yaw[self.sdc_id, 0] = new_vel_yaw
        reward = 0
        # update done
        if self.init_timestep == self.max_timestep:
            done = True
        elif len(self.collision_hist) != 0:
            done = True
            reward = -1000
        else:
            done = False
        # still need to set the task reward
        done = torch.tensor(done, dtype=torch.bool)
        # update tensordict
        out = TensorDict(
            {
                "next": {
                    "x": torch.tensor(x, dtype=torch.float32),
                    "y": torch.tensor(y, dtype=torch.float32),
                    "z": torch.tensor(z, dtype=torch.float32),
                    "heading_angle": torch.tensor(heading_angle, dtype=torch.float32),
                    "velocity_x": torch.tensor(velocity_x, dtype=torch.float32),
                    "velocity_y": torch.tensor(velocity_y, dtype=torch.float32),
                    "vel_yaw": torch.tensor(vel_yaw, dtype=torch.float32),
                    "time_step": torch.tensor(self.init_timestep, dtype=torch.float32),
                    # "ase_latent"
                    "done": done,
                    "reward": reward
                }
            },
            tensordict.shape,
        )
        # calculate reward
        state = tensordict["state"]
        next_state = out["observation"]
        # update carla env
        ids = self.ids.clone()
        for actor, actor_id in self.actor_list:
            x, y, z, heading_angle, velocity_x, velocity_y, vel_yaw, length, width, height, timestamp_micros, is_self_driving, valid = self.load_state(
                self.init_timestep, actor_id)
            if valid == 1:
                actor.set_transform(carla.Transform(
                    carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=self.from_radian_to_degree(init_heading_angle))))
                actor.set_velocity(carla.Vector3D(
                    x=velocity_x, y=velocity_y, z=0))
                actor.set_angular_velocity(
                    carla.Vector3D(x=0, y=0, z=vel_yaw))
            else:
                actor.destroy()
                self.actor_list.remove((actor, actor_id))
            ids.remove(actor_id)
        for other_id in ids:
            x, y, z, heading_angle, velocity_x, velocity_y, vel_yaw, length, width, height, timestamp_micros, is_self_driving, valid = self.load_state(
                self.init_timestep, other_id)
            if valid == 1:
                object_type = self.object_type[other_id]
                self.create_actor(object_type, other_id, x, y, z, heading_angle,
                                  velocity_x, velocity_y, vel_yaw, length, width, height, timestamp_micros, is_self_driving)
        return out

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _make_spec(self, td_params):
        self.action_spec = CompositeSpec(
            BoundedTensorSpec(shape=(1,), dtype=torch.float32,
                              minimum=0, maximum=1.0, name="throttle"),
            BoundedTensorSpec(shape=(1,), dtype=torch.float32,
                              minimum=-1.0, maximum=1.0, name="steer"),
            BoundedTensorSpec(shape=(1,), dtype=torch.float32,
                              minimum=0.0, maximum=1.0, name="brake"),
        )
        self.state_spec = CompositeSpec(
            UnboundedContinuousTensorSpec(shape=(self.object_num, 1), dtype=torch.float32,
                                          name="x"),
            UnboundedContinuousTensorSpec(shape=(self.object_num, 1), dtype=torch.float32,
                                          name="y"),
            UnboundedContinuousTensorSpec(shape=(self.object_num, 1), dtype=torch.float32,
                                          name="z"),
            BoundedTensorSpec(shape=(self.object_num, 1), dtype=torch.float32,
                              minimum=0, maximum=2*torch.pi, name="heading_angle"),
            UnboundedContinuousTensorSpec(shape=(self.object_num, 1), dtype=torch.float32,
                                          name="velocity_x"),
            UnboundedContinuousTensorSpec(shape=(self.object_num, 1), dtype=torch.float32,
                                          name="velocity_y"),
            UnboundedContinuousTensorSpec(shape=(self.object_num, 1), dtype=torch.float32,
                                          name="vel_yaw"),
            UnboundedDiscreteTensorSpec(shape=(self.object_num, 1), dtype=torch.int64,
                                        name="time_step"),
            # UnboundedContinuousTensorSpec(shape=(self.object_num, 1), dtype=torch.float32,
            #                              name="length"),
            # UnboundedContinuousTensorSpec(shape=(self.object_num, 1), dtype=torch.float32,
            #                              name="width"),
            # UnboundedContinuousTensorSpec(shape=(self.object_num, 1), dtype=torch.float32,
            #                              name="height"),
            # UnboundedContinuousTensorSpec(shape=(self.object_num, 1), dtype=torch.float32,
            #                              name="timestamp_micros"),
            # BinaryTensorSpec(shape=(self.object_num, 1), dtype=torch.int64,
            #                 name="is_self_driving"),
        )
        self.observation_spec = self.state_spec.clone()
        self.done_spec = BinaryTensorSpec(shape=(1,), dtype=torch.int64,
                                          name="done")
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(1,), dtype=torch.float32, name="reward")

    def find_match_vehicles(self, length, width, height):
        closest_match = None
        smallest_difference = float('inf')

        target_length = length
        target_width = width
        target_height = height

        for vehicle in self.blueprint_library.filter('vehicle.*'):
            length = blueprint.get_attribute('length').as_float()
            width = blueprint.get_attribute('width').as_float()
            height = blueprint.get_attribute('height').as_float()

            difference = math.sqrt((length - target_length)**2 +
                                   (width - target_width)**2 + (height - target_height)**2)

            # If this difference is the smallest so far, update the closest match
            if difference < smallest_difference:
                closest_match = blueprint
                smallest_difference = difference

        return closest_match


if __name__ == "__main__":
    env = CarlaEnv()
    check_env_specs(env)
