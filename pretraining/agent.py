import torch
import torch.nn as nn
from collections import defaultdict
from enc_dic import Encoder, Discriminator, Mlp
import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
frame_skip = 1
frames_per_batch = 91 // frame_skip
# For a complete training, bring the number of frames up to 1M
total_frames = 100000 // frame_skip


class Agent():
    def __init__(state_size, action_size, env):
        self.action_distribution_variance = 0.0025
        self.skill_discovery_weight = 0.5
        self.gradient_penalty_weight = 5
        self.diversity_weight = 0.01
        self.encoder_scaling_factor = 1
        self.encoder_coefficient = 5
        self.disc_coefficient = 5
        self.episode_length = 91
        self.units = [1024, 1024, 512]
        self.ase_latent_shape = [91, 64]
        self.encoder = Encoder(self.ase_latent_shape)
        self.discriminator = Discriminator()
        self.mlp = Mlp(state_size)
        self.actor_net = nn.Sequential(
            nn.Linear(state_size, self.units[0]),
            nn.ReLU(),
            nn.Linear(self.units[0], self.units[1]),
            nn.ReLU(),
            nn.Linear(self.units[1], self.units[2]),
            nn.ReLU(),
            nn.Linear(self.units[2], 2 * self.action_size),
            nn.Tanh(),
            NormalParamExtractor(),
        )
        self.policy_module = ProbabilisticActor(
            module=TensorDictModule(
                actor_net, in_keys=["observation"], out_keys=["mu", "sigma"]
            ),
            spec=env.action_spec,
            in_keys=["mu", "sigma"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "min": env.action_spec.space.minimum,
                "max": env.action_spec.space.maximum,
            },
            return_log_prob=True,
            # we'll need the log-prob for the numerator of the importance weights
        )
        self.value_net = nn.Sequential(
            nn.Linear(self.state_size, self.units[0]),
            nn.ReLU(),
            nn.Linear(self.units[0], self.units[1]),
            nn.ReLU(),
            nn.Linear(self.units[1], self.units[2]),
            nn.ReLU(),
            nn.Linear(self.units[2], 1),
            nn.Tanh()
        )
        self.value_module = ValueOperator(
            module=value_net,
            in_keys=["observation"],
            out_keys=["value"],
        )
        # need to revise the transformedEnv
        env = TransformedEnv(
            env,
            Compose(
                [
                    CatTensors(
                        in_keys=["observation", "ase_latent"], out_key="observation"),
                    DoubleToFloat(),
                    ObservationNorm(
                        mean=env.observation_spec.space.mean,
                        std=env.observation_spec.space.std,
                    ),
                    StepCounter(),
                ]
            ),
        )
        self.collector = SyncDataCollector(
            env,
            self.policy_module,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            split_trajs=False,
            device=device,
        )
        # clip value for PPO loss: see the equation in the intro for more context.
        clip_epsilon = 0.2
        gamma = 0.99
        lmbda = 0.95
        entropy_eps = 1e-4
        self.advantage_module = GAE(
            gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
        )
        self.loss_module = ClipPPOLoss(
            actor=self.policy_module,
            critic=self.value_module,
            clip_epsilon=clip_epsilon,
            entropy_bonus=bool(entropy_eps),
            entropy_coef=entropy_eps,
            # these keys match by default but we set this for completeness
            value_target_key=advantage_module.value_target_key,
            critic_coef=1.0,
            gamma=0.99,
            loss_critic_type="smooth_l1",
        )
        check_env_specs(env)
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(frames_per_batch),
            sampler=SamplerWithoutReplacement(),
        )
        self.optim = torch.optim.Adam(loss_module.parameters(), lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, total_frames // frames_per_batch, 0.0
        )

    def train():
        logs = defaultdict(list)
        pbar = tqdm(total=total_frames * frame_skip)
        eval_str = ""
        sub_batch_size = 64
        for i, tensordict_data in enumerate(self.collector):
            for _ in range(num_epoch):
                with torch.no_grad():
                    # need revise, z should be a sequence of latents {z_1,z_2...z_T}
                    z = self.sample_latents(self.episode_length)
                    # z has shape [batch_size, episode_length, latent_size]
                    # split tensordict_data's reward across first dimension
                    combine_rewards = self.combine_rewards(
                        tensordict_data["observsation"], tensordict_data["next"]["observation"], z
                    )
                    # the shape of reward is [batch_size, 3]
                    print(combine_rewards.shape)
                    tensordict_data["next"]["reward"] += combine_rewards
                    advantage = self.compute_advantage(
                        tensordict_data["observation"],
                        tensordict_data["reward"],
                        tensordict_data["done"],
                        tensordict_data["next"]["observation"],
                    )
                data_view = tensordict_data.reshape(-1)
                replay_buffer.extend(data_view.cpu())
                for _ in range(frames_per_batch // sub_batch_size):
                    subdata = replay_buffer.sample(sub_batch_size)
                    loss_vals = loss_module(subdata)
                    agent_state = subdata["observation"]
                    agent_next_state = subdata["next"]["observation"]
                    # calculate the encoder loss
                    encoder_pred = self.eval_encoder(
                        agent_state, agent_next_state)
                    encoder_loss = self._enc_loss(encoder_pred, z)
                    # calculate the discriminator loss
                    discriminator_loss = None
                    diversity_loss = None

                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                        + encoder_loss * self.encoder_coefficient
                        + discriminator_loss * self.disc_coefficient
                        + diversity_loss
                    )

                    # Optimization: backward, grad clipping and optim step
                    loss_value.backward()
                    # this is not strictly mandatory but it's good practice to keep
                    # your gradient norm bounded
                    torch.nn.utils.clip_grad_norm_(
                        loss_module.parameters(), max_grad_norm)
                    optim.step()
                    optim.zero_grad()
        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel() * frame_skip)
        cum_reward_str = (
            f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
        if i % 10 == 0:
            # We evaluate the policy once every 10 batches of data.
            # Evaluation is rather simple: execute the policy without exploration
            # (take the expected value of the action distribution) for a given
            # number of steps (1000, which is our env horizon).
            # The ``rollout`` method of the env can take a policy as argument:
            # it will then execute this policy at each step.
            with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
                # execute a rollout with the trained policy
                eval_rollout = env.rollout(1000, policy_module)
                logs["eval reward"].append(
                    eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                logs["eval step_count"].append(
                    eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}"
                )
                del eval_rollout
        pbar.set_description(
            ", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()

    def visualization(self, logs):
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.plot(logs["reward"])
        plt.title("training rewards (average)")
        plt.subplot(2, 2, 2)
        plt.plot(logs["step_count"])
        plt.title("Max step count (training)")
        plt.subplot(2, 2, 3)
        plt.plot(logs["eval reward (sum)"])
        plt.title("Return (test)")
        plt.subplot(2, 2, 4)
        plt.plot(logs["eval step_count"])
        plt.title("Max step count (test)")
        plt.show()

    def compute_advantage(self, state, reward, done, next_state):
        advantage, value_target = self.advantage_module(
            obs=obs, reward=reward, done=done, next_obs=next_obs)

    def sample_latents():
        return self.model.sample_latents(self.episode_length)

    def eval_encoder(self, state, next_state):
        obs = torch.cat([state, next_state], dim=0)
        enc_mlp_out = self.mlp(obs)
        enc_out = self.encoder(enc_mlp_out)
        enc_output = torch.nn.functional.normalize(enc_output, dim=-1)
        return enc_out

    def eval_discriminator(self, state, next_state):
        obs = torch.cat([state, next_state], dim=0)
        disc_mlp_out = self.mlp(obs)
        disc_out = self.discriminator(disc_mlp_out)
        return disc_out

    def get_disc_weights(self):
        weights = []
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                weights.append(torch.flatten(m.weight))

        weights.append(torch.flatten(self.discriminator.weight))
        return weights

    def get_enc_weights(self):
        weights = []
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                weights.append(torch.flatten(m.weight))

        weights.append(torch.flatten(self.encoder.weight))
        return weights

    def _calc_enc_error(self, enc_pred, ase_latent):
        err = enc_pred * ase_latent
        err = -torch.sum(err, dim=-1, keepdim=True)
        return err

    def _enc_loss(self, enc_pred, ase_latent):
        # enc_coef: 5
        # enc_weight_decay: 0.0000
        # enc_reward_scale: 1
        # enc_grad_penalty: 0
        enc_error = self._calc_enc_error(enc_pred, ase_latent)
        enc_loss = torch.mean(enc_err)
        return enc_loss

    def _calc_enc_rewards(self, state, next_state, ase_latents):
        with torch.no_grad():
            enc_pred = self.eval_encoder(state, next_state)
            err = self._calc_enc_error(enc_pred, ase_latents)
            enc_r = torch.clamp_min(-err, 0.0)
            enc_r *= self._enc_reward_scale

        return enc_r

    def _calc_disc_rewards(self, state, next_state):
        with torch.no_grad():
            disc_logits = self.eval_discriminator(state, next_state)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob,
                                torch.tensor(0.0001)))
            disc_r *= self._disc_reward_scale

        return disc_r

    def _disc_loss_neg(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.zeros_like(disc_logits))
        return loss

    def _disc_loss_pos(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.ones_like(disc_logits))
        return loss

    def _disc_loss(self, disc_agent_logit, disc_demo_logit, obs_demo):
        self.disc_coef = 5
        self.disc_logit_reg = 0.01
        self.disc_grad_penalty = 5
        self.disc_reward_scale = 2
        self.disc_weight_decay = 0.0001
        # prediction loss
        disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
        disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # logit reg
        logit_weights = self.get_disc_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss += self.disc_logit_reg * disc_logit_loss

        # grad penalty
        disc_demo_grad = torch.autograd.grad(disc_demo_logit, obs_demo, grad_outputs=torch.ones_like(disc_demo_logit),
                                             create_graph=True, retain_graph=True, only_inputs=True)
        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_demo_grad)
        disc_loss += self.disc_grad_penalty * disc_grad_penalty

        # weight decay
        if (self.disc_weight_decay != 0):
            disc_weights = self.model.a2c_network.get_disc_weights()
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss += self._disc_weight_decay * disc_weight_decay

        disc_agent_acc, disc_demo_acc = self._compute_disc_acc(
            disc_agent_logit, disc_demo_logit)

        disc_info = {
            'disc_loss': disc_loss,
            'disc_grad_penalty': disc_grad_penalty.detach(),
            'disc_logit_loss': disc_logit_loss.detach(),
            'disc_agent_acc': disc_agent_acc.detach(),
            'disc_demo_acc': disc_demo_acc.detach(),
            'disc_agent_logit': disc_agent_logit.detach(),
            'disc_demo_logit': disc_demo_logit.detach()
        }
        return disc_info

    def _diversity_loss(self, obs, action_params, ase_latents):
        assert (self.model.a2c_network.is_continuous)

        n = obs.shape[0]
        assert (n == action_params.shape[0])

        new_z = self._sample_latents(n)
        mu, sigma = self.policy_module(obs=obs, ase_latents=new_z)

        clipped_action_params = torch.clamp(action_params, -1.0, 1.0)
        clipped_mu = torch.clamp(mu, -1.0, 1.0)

        a_diff = clipped_action_params - clipped_mu
        a_diff = torch.mean(torch.square(a_diff), dim=-1)

        z_diff = new_z * ase_latents
        z_diff = torch.sum(z_diff, dim=-1)
        z_diff = 0.5 - 0.5 * z_diff

        diversity_bonus = a_diff / (z_diff + 1e-5)
        diversity_loss = torch.square(
            self._amp_diversity_tar - diversity_bonus)

        if self._amp_diversity_bonus != 0:
            diversity_loss = (1+self._amp_diversity_bonus) * diversity_loss
        return diversity_loss

    def combine_rewards(self, state, next_state, ase_latents):
        self._disc_reward_w = 0.5
        self._enc_reward_w = 0.5
        disc_r = self._calc_disc_rewards(state, next_state)
        enc_r = self._calc_enc_rewards(state, next_state, ase_latents)
        combined_rewards = self._disc_reward_w * disc_r \
            + self._enc_reward_w * enc_r
        return combined_rewards

    def _sample_latents(self, n):
        z = torch.normal(torch.zeros([n, self._ase_latent_shape[-1]]))
        z = torch.nn.functional.normalize(z, dim=-1)
        return z
