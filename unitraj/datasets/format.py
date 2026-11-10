"""UniTraj-native feature conversion without added training labels."""
import numpy as np
import torch

class FeatureConverter:
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def convert_to_fmae_format(self, input):
        B = len(input['center_objects_id']) #number of interested agents/scenarios
        N_agent = self.config["max_num_agents"]#10 #maximum number of agents per scenari --> get from config file
        N_lane = self.config["max_num_roads"] #80 #maximum number of lane segments per scenario --> get from config file
        h_steps = self.config["past_len"]#50 #num historical timesteps for agent tracks --> get from config file
        f_steps = self.config["future_len"] #60 #num future timesteps for agent tracks --> get from config file
        lane_sampling_pts = self.config["max_points_per_lane"]
        data = {
            "x": torch.rand(B, N_agent, h_steps, 2), #agent tracks as local differences from origin being the last position of the past trajectory
            "x_attr": torch.zeros((B, N_agent, 3), dtype=torch.uint8), #categorical agent attributes
            "x_positions": torch.rand(B, N_agent, h_steps, 2), #agent tracks in scene coordinates
            "x_centers": torch.rand(B, N_agent, 2), #center of agent track
            "x_angles": torch.rand(B, N_agent, h_steps+f_steps), #agent headings
            "x_velocity": torch.rand(B, N_agent, h_steps+f_steps), #velocity of agents as absolute values
            "x_velocity_diff": torch.rand(B, N_agent, h_steps), #velocity changes of agents
            "lane_positions": torch.rand(B, N_lane, lane_sampling_pts, 2), #lane segments in scene coordinates
            "lane_centers": torch.rand(B, N_lane, 2), #center of lane segments
            "lane_angles": torch.rand(B, N_lane), #orientation of lane segments
            "lane_attr": torch.rand(B, N_lane, 3), #categorial lane attributes
            "is_intersections": torch.rand(B, N_lane), # categorical lane attribute
            "y": torch.rand(B, N_agent, f_steps, 2), #agent future tracks as x,y positions -> ground truth in unitraj format
            "x_padding_mask": torch.zeros((B, N_agent, h_steps+f_steps), dtype=torch.bool), #padding mask for agent tracks
            "lane_padding_mask": torch.zeros((B, N_lane, lane_sampling_pts), dtype=torch.bool), #padding mask for lane segment points
            "x_key_padding_mask": torch.zeros((B, N_agent), dtype=torch.bool), #batch padding mask for agent tracks
            "lane_key_padding_mask": torch.zeros((B, N_lane), dtype=torch.bool), #batch padding mask for lane segments
            "num_actors": torch.full((B,), fill_value=N_agent, dtype=torch.int64),
            "num_lanes": torch.full((B,), fill_value=N_lane, dtype=torch.int64),
            "scenario_id": [] * B,
            "track_id": [] * B,
            "origin": torch.rand(B, 2), #scene to global coordinates position
            "theta": torch.rand(B), #scene to global coordinates orientation
        }
        for key, value in input.items():
            if isinstance(value, torch.Tensor):
                    input[key] = input[key].to("cpu")

        data["origin"] = input["center_objects_world"][..., 0:2].clone()
        data["theta"] = input["center_objects_world"][..., 6].clone()

        data["x"][..., 1:h_steps, 0:2] = torch.where(
            (~input["obj_trajs_mask"][..., :(h_steps - 1)] | ~input["obj_trajs_mask"][..., 1:h_steps]).unsqueeze(-1),
            torch.zeros(B, N_agent, h_steps - 1, 2),
            input["obj_trajs_pos"][..., 1:h_steps, 0:2] - input["obj_trajs_pos"][..., :(h_steps - 1), 0:2],
        )
        data["x"][... , 0, 0:2] = torch.zeros(B, N_agent, 2)

        #data["x_attr"] -> object type | object category | object type combined -> only third attribute is used in model
        data["x_attr"][:, :, 0] = 9 #unknown
        for center_idx, center_obj in enumerate(input["obj_trajs"]):
            for actor_idx, actor in enumerate(center_obj):
                if actor[h_steps - 1][6] == 1:
                    data["x_attr"][center_idx, actor_idx, 0] = 0 #vehicle
                    data["x_attr"][center_idx, actor_idx, 2] = 0 
                elif actor[h_steps - 1][7] == 1:
                    data["x_attr"][center_idx, actor_idx, 0] = 1 #pedestrian
                    data["x_attr"][center_idx, actor_idx, 2] = 1
                elif actor[h_steps - 1][8] == 1:
                    data["x_attr"][center_idx, actor_idx, 0] = 3 #bicycle
                    data["x_attr"][center_idx, actor_idx, 2] = 2
                else:
                    data["x_attr"][center_idx, actor_idx, 0] = 9 #unknown
                    data["x_attr"][center_idx, actor_idx, 2] = 3

                if actor[h_steps - 1][9] == 1:
                    data["x_attr"][center_idx, actor_idx, 1] = 3 #FOCAL_TRACK -> track that is being predicted
                if actor[h_steps - 1][10] == 1:
                    data["x_attr"][center_idx, actor_idx, 1] = 3 #FOCAL_TRACK -> track that is being predicted & is the sdc track
                if actor[h_steps - 1][9] == 0 and actor[0][10] == 0:
                    data["x_attr"][center_idx, actor_idx, 1] = 2 #SCORED_TRACK -> track that is being scored
        data["x_positions"] = input["obj_trajs"][..., :h_steps, :2].clone()
        data["x_centers"] = input["obj_trajs"][..., h_steps - 1, :2].clone() 
        data["x_angles"] = np.arcsin(input["obj_trajs"][..., h_steps+12].float())
        data["x_velocity"] = torch.cat(
            (torch.from_numpy(np.linalg.norm(input["obj_trajs"][..., h_steps+14:h_steps+16], axis=-1)), 
             torch.from_numpy(np.linalg.norm(input["obj_trajs_future_state"][..., 2:], axis=-1))), 
             dim=-1)
        data["lane_positions"] = input["map_polylines"][..., :2].clone()
        data["lane_centers"] = data["lane_positions"][:, :, ((lane_sampling_pts//2)-1):(lane_sampling_pts//2)+1].mean(dim=-2)
        data["lane_angles"] = torch.atan2(
             data["lane_positions"][..., lane_sampling_pts//2, 1] -  data["lane_positions"][..., (lane_sampling_pts//2)-1, 1],
             data["lane_positions"][..., lane_sampling_pts//2, 0] -  data["lane_positions"][..., (lane_sampling_pts//2)-1, 0],
        )

        #data["lane_attr"] -> not used in model
        #data["is_intersections"] -> not used in model

        data["y"] = torch.where(
            (~input["obj_trajs_mask"][..., (h_steps - 1)].unsqueeze(-1) | ~input["obj_trajs_future_mask"][..., :]).unsqueeze(-1),
            torch.zeros(B, N_agent, f_steps, 2),
            input["obj_trajs_future_state"][..., :, 0:2] - input["obj_trajs"][..., (h_steps - 1), 0:2].unsqueeze(-2),
        )
        data["x_padding_mask"] = torch.cat((~input["obj_trajs_mask"], ~input["obj_trajs_future_mask"]), dim=-1)
        data["lane_padding_mask"] = ~input["map_polylines_mask"].clone()
        data["lane_key_padding_mask"] = data["lane_padding_mask"].all(-1)
        data["num_actors"] =  (~data["x_key_padding_mask"]).sum(-1)
        data["num_lanes"] = (~data["lane_key_padding_mask"]).sum(-1)
        data["scenario_id"] = input["scenario_id"]

        data["x_padding_mask"] = torch.where(
            torch.from_numpy(np.linalg.norm(data["x_positions"][..., h_steps - 1, :], axis=-1) > 150).unsqueeze(-1),
            torch.ones_like(data["x_padding_mask"]),
            data["x_padding_mask"]
        )
        data["x_padding_mask"] = torch.where(
            (data["x_padding_mask"][..., h_steps - 1]).unsqueeze(-1),
            torch.ones_like(data["x_padding_mask"]),
            data["x_padding_mask"]
        )
        data["x_key_padding_mask"] = data["x_padding_mask"].all(-1)

        data["x_velocity_diff"][..., 1:h_steps] = torch.where(
            (~input["obj_trajs_mask"][..., :(h_steps - 1)] | ~input["obj_trajs_mask"][..., 1:h_steps]),
            torch.zeros(B, N_agent, h_steps-1),
            data["x_velocity"][..., 1:h_steps] - data["x_velocity"][..., :(h_steps - 1)],
        )
        data["x_velocity_diff"][..., 0] = torch.zeros(N_agent)

        #data["track_id"] information not present in input -> only used for av2 submission

        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(self.device)
        return data
