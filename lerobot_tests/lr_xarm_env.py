import datetime
from lerobot.common.policies.tdmpc.modeling_tdmpc import TDMPCConfig, TDMPCPolicy
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import torch
from torch.utils.tensorboard import SummaryWriter
import time

device = "cuda"
training_steps = 10000

if __name__ == "__main__":

    

    dataset_metadata = LeRobotDatasetMetadata("lerobot/xarm_lift_medium")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # Create TDMPC config
    cfg = TDMPCConfig(
        input_features=input_features,
        output_features=output_features,
        # use_mpc=True,
    )

    delta_timestamps = {
        "observation.image": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
        "next.reward": [i / dataset_metadata.fps for i in cfg.reward_delta_indices],
    }
    dataset = LeRobotDataset('lerobot/xarm_lift_medium', delta_timestamps=delta_timestamps)

    policy = TDMPCPolicy(cfg, dataset_stats=dataset.meta.stats)
    policy.train()
    policy.to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=8,
        batch_size=32,
        shuffle=True,
        # pin_memory=device != torch.device("cpu"),
        drop_last=True,
    )

    curr_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    writer = SummaryWriter(log_dir=f"runs/xarm_tdmpc/{curr_time}")

    step = 0
    done = False
    while not done:
        end = time.time()
        for batch in dataloader:
            data_load_time = time.time()
            # batch["observation.state"] = torch.cat([batch["observation.state.qpos"], batch["observation.state.gripper"].unsqueeze(2)], 2)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k not in ['task']}
            gpu_load_time = time.time()
            output_dict = policy.forward(batch)
            pred_time = time.time()
            
            loss, _ = output_dict
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_time = time.time()


            writer.add_scalar("Loss/train", loss.item(), step)
            writer.add_scalar("Time/data_load", data_load_time - end, step)
            writer.add_scalar("Time/gpu_transfer", gpu_load_time - data_load_time, step)
            writer.add_scalar("Time/pred_time", pred_time - gpu_load_time, step)
            writer.add_scalar("Time/train_time", train_time - pred_time, step)
            writer.add_scalar("Time/step_time", time.time() - end, step) 

            if step % 250 == 0:
                print(f"step: {step} loss: {loss.item():.3f}")

            step += 1
            if step >= training_steps:
                done = True
                break

            if step % 1000 == 0:
                policy.save_pretrained(f"ckpts/xarm_tdmpc_step_{step}")
            
            end = time.time()
            # dataset.hf_dataset.clear_cache()
            
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')  # Get statistics by line number
    # for stat in top_stats[:10]:  # Show top 10 memory allocations
    #     print(stat)
    # tracemalloc.stop()  # Stop tracing
    policy.save_pretrained(f"ckpts/xarm_tdmpc_step_{step}")
