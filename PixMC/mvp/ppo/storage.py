#!/usr/bin/env python3

"""Rollout storage."""

import torch

from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.sampler import SubsetRandomSampler

from mvp.ppo.bert import BERT, MAE
from mvp.ppo.icm import ICM
from mvp.ppo.rnd import RND
from mvp.ppo.ngu import NGU
from mvp.ppo.bert_ratio import BERT_RATIO

class RolloutStorage:

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        states_shape,
        actions_shape,
        expl_cfg,
        device="cpu",
        sampler="sequential"
    ):

        self.device = device
        self.sampler = sampler

        # Core
        self.observations = None
        self.states = torch.zeros(num_transitions_per_env, num_envs, *states_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.explore = (expl_cfg["expl_seq_len"] >= 1)
        self.baseline = expl_cfg["baseline"]
        self.use_my_ratio = expl_cfg["use_my_ratio"]
        self.expl_seq_len_ratio = expl_cfg["expl_seq_len_ratio"]
        # self.k_expl_ratio = expl_cfg["k_expl_ratio"]
        if self.use_my_ratio:
            self.seq_obs_ratio = torch.zeros(
                num_transitions_per_env, num_envs, self.expl_seq_len_ratio,
                expl_cfg["input_dim"], device=self.device
            )
            self.seq_next_obs_ratio = torch.zeros(
                num_transitions_per_env, num_envs, self.expl_seq_len_ratio,
                expl_cfg["input_dim"], device=self.device
            )
            self.seq_act_ratio = torch.zeros(
                num_transitions_per_env, num_envs, self.expl_seq_len_ratio,
                *actions_shape, device=self.device
            )
        if self.explore:
            # Optional exploration bonus
            self.expl_r = torch.zeros_like(self.rewards, device=self.device)
            self.k_expl = expl_cfg["k_expl"]
            self.expl_seq_len = expl_cfg["expl_seq_len"]
            self.seq_obs = torch.zeros(
                num_transitions_per_env, num_envs, self.expl_seq_len,
                expl_cfg["input_dim"], device=self.device
            )
            self.seq_next_obs = torch.zeros(
                num_transitions_per_env, num_envs, self.expl_seq_len,
                expl_cfg["input_dim"], device=self.device
            )
            self.seq_act = torch.zeros(
                num_transitions_per_env, num_envs, self.expl_seq_len,
                *actions_shape, device=self.device
            )

            if expl_cfg["baseline"] == "icm":
                # NOTE: assuming action has only 1 dim
                self.icm = ICM(
                    expl_cfg["input_dim"], actions_shape[-1]).to(self.device)
            elif expl_cfg["baseline"] == "rnd":
                self.rnd = RND(expl_cfg["input_dim"]).to(self.device)
            elif expl_cfg["baseline"] == "ngu":
                self.ngu = NGU(
                    expl_cfg["input_dim"], actions_shape[-1]).to(self.device)
            elif expl_cfg["baseline"] == "avg":
                self.avg_iter = 0
                self.obs_avg = torch.zeros(
                    self.expl_seq_len, expl_cfg["input_dim"],
                    device=self.device)
            else:
                self.n_mask = expl_cfg["n_mask"]

                if expl_cfg["mask_all"]:
                    self.bert = MAE(
                        seq_len=self.expl_seq_len,
                        feature_dim=expl_cfg["input_dim"],
                        embed_dim=expl_cfg["embed_dim"],
                        decoder_embed_dim=expl_cfg["decoder_embed_dim"],
                        decoder_num_heads=expl_cfg["decoder_num_heads"],
                        decoder_depth=expl_cfg["decoder_depth"],
                        mask_ratio=expl_cfg["mask_ratio"],
                        norm_loss=expl_cfg["norm_loss"],
                        use_cls=expl_cfg["use_cls"],
                        ).to(self.device)
                else:
                    self.bert = BERT(
                        seq_len=self.expl_seq_len,
                        feature_dim=expl_cfg["input_dim"],
                        embed_dim=expl_cfg["embed_dim"],
                        decoder_embed_dim=expl_cfg["decoder_embed_dim"],
                        decoder_num_heads=expl_cfg["decoder_num_heads"],
                        decoder_depth=expl_cfg["decoder_depth"],
                        mask_ratio=expl_cfg["mask_ratio"],
                        norm_loss=expl_cfg["norm_loss"],
                        use_cls=expl_cfg["use_cls"],
                        ).to(self.device)
                    #  ----------------------------------------------------------------
                self.bert_opt = torch.optim.Adam(
                    self.bert.parameters(), lr=expl_cfg["bert_lr"])
    # print(action_shape)
        if self.use_my_ratio:
            self.bert_ratio = BERT_RATIO(
                seq_len=self.expl_seq_len_ratio,
                feature_dim=expl_cfg["input_dim"],
                embed_dim=expl_cfg["embed_dim"],
                decoder_embed_dim=expl_cfg["decoder_embed_dim"],
                decoder_num_heads=expl_cfg["decoder_num_heads"],
                decoder_depth=expl_cfg["decoder_depth_ratio"],
                mask_ratio=expl_cfg["mask_ratio"],
                norm_loss=expl_cfg["norm_loss_ratio"],
                use_cls=expl_cfg["use_cls_ratio"],
                action_shape=actions_shape[-1],
                attn_selet_way=expl_cfg["attn_selet_way"],
                sequencelevel=expl_cfg["sequencelevel"],
                ratio_s2aanda2s=expl_cfg["ratio_s2aanda2s"],
                attn_ratio_weight=expl_cfg["attn_ratio_weight"]
                ).to(device)
#  ----------------------------------------------------------------
            self.bert_ratio_opt = torch.optim.Adam(
                self.bert_ratio.parameters(), lr=expl_cfg["bert_lr_ratio"])

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.sequencelevel = expl_cfg["sequencelevel"]
        self.batch_size = self.num_envs * self.num_transitions_per_env
        self.episodic_obs_emb_history = [None for _ in range(self.num_envs)]

        self.step = 0

    def add_transitions(
        self, observations, states, actions, rewards, dones, values,
        actions_log_prob, mu, sigma, seq_obs, seq_act, seq_next_obs, seq_obs_ratio, seq_act_ratio, seq_next_obs_ratio):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        if self.observations is None:
            self.observations = torch.zeros(
                self.num_transitions_per_env, self.num_envs,
                *observations.shape[1:], device=self.device
            )
        self.observations[self.step].copy_(observations)
        self.states[self.step].copy_(states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1))
        self.values[self.step].copy_(values)
        self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(mu)
        self.sigma[self.step].copy_(sigma)

        # for env_id in range(self.num_envs):
        #     if dones[env_id]:
        #         self.episodic_obs_emb_history[env_id] = None

        if self.explore:
            if self.expl_seq_len == 1:
                self.seq_obs[self.step].copy_(seq_obs)
            else:
                self.seq_obs[self.step].copy_(seq_obs)
                self.seq_next_obs[self.step].copy_(seq_next_obs)
                self.seq_act[self.step].copy_(seq_act)
        if self.use_my_ratio:
            self.seq_obs_ratio[self.step].copy_(seq_obs_ratio)
            self.seq_next_obs_ratio[self.step].copy_(seq_next_obs_ratio)
            self.seq_act_ratio[self.step].copy_(seq_act_ratio)

        self.step += 1

    def clear(self):
        self.step = 0

    def compute_expl_r(self, anneal_weight=1.0):
        if not self.explore:
            return 0.
        if self.use_my_ratio:
            bert_ratio = self.get_expl_ratio()
        else:
            bert_ratio = 1
        # mean_bert_ratio = bert_ratio

        M, N, T, D = self.seq_obs.shape

        if self.baseline == "rnd":
            # compute intrinsic reward using RND
            obs = self.seq_obs[:, :, 0].view(M * N, D)
            rnd_loss = self.rnd(obs) # rnd_loss:[16384]
            self.expl_r = rnd_loss.view(*self.rewards.shape) # expl_r:[32, 512, 1]
            if self.use_my_ratio:
                if self.sequencelevel:
                    bert_ratio = torch.mean(bert_ratio, dim=1, keepdim=True)
                bert_ratio = bert_ratio.view(*self.rewards.shape).detach()
            else:
                bert_ratio = 1
            self.expl_r = self.expl_r.detach() * bert_ratio
        elif self.baseline == "icm":
            obs = self.seq_obs[:, :, 0].view(M * N, D)
            next_obs = self.seq_obs[:, :, 1].view(M * N, D)
            actions = self.actions.view(M * N, -1)  # (M, N, action_dim)

            icm_loss = self.icm(obs, next_obs, actions)  # (M*N,)

            # relabel rewards
            self.expl_r = icm_loss.view(*self.rewards.shape)
            if self.use_my_ratio:
                if self.sequencelevel:
                    bert_ratio = torch.mean(bert_ratio, dim=1, keepdim=True)
                bert_ratio = bert_ratio.view(*self.rewards.shape).detach()
            else:
                bert_ratio = 1
            self.expl_r = self.expl_r.detach() * bert_ratio
        elif self.baseline == "ngu":
            # obs = self.seq_obs[:, :, 0].view(M * N, D)
            # next_obs = self.seq_obs[:, :, 1].view(M * N, D)
            # actions = self.actions.view(M * N, -1)  # (M, N, action_dim)

            obs = self.seq_obs[:, :, 0]
            next_obs = self.seq_obs[:, :, 1]
            actions = self.actions  # (M, N, action_dim)

            # self.episodic_obs_emb_history = [None for _ in range(self.batch_size)]
            ngu_loss = self.ngu(obs, next_obs, actions, self.episodic_obs_emb_history, self.dones)  # (M*N,)
            ngu_loss = torch.from_numpy(ngu_loss).float().to(self.device)
            # relabel rewards
            self.expl_r = ngu_loss.view(*self.rewards.shape)
            if self.use_my_ratio:
                if self.sequencelevel:
                    bert_ratio = torch.mean(bert_ratio, dim=1, keepdim=True)
                bert_ratio = bert_ratio.view(*self.rewards.shape).detach()
            else:
                bert_ratio = 1
            self.expl_r = self.expl_r.detach() * bert_ratio
        elif self.baseline == "avg":
            assert self.baseline is not "avg"
            # calculate exploration reward from moving average
            # (M, N, T, D) => (M, N, 1)
            self.expl_r = ((self.seq_obs - self.obs_avg) ** 2).mean(
                dim=-1).mean(dim=-1, keepdim=True)

            # update moving average of obs embeddings
            self.avg_iter += 1
            avg = self.seq_obs.view(M * N, T, D).mean(dim=0)
            self.obs_avg = self.obs_avg - (self.obs_avg - avg) / self.avg_iter
        else:
            bert_input = self.seq_obs.view(M * N, T, D) # self.seq_obs:[32, 512, 5, 384], bert_input:[16384, 5, 384]
            # calculate BERT loss
            # (M, N, T, *obs_shape)
            bert_loss, _, mask, loss_reward = self.bert(bert_input, keep_batch=True)
            if self.n_mask > 1:
                assert self.n_mask <= 1
                # mask multiple times and calculate average to reduce variance
                for _ in range(self.n_mask - 1):
                    l, _, mask, loss_reward = self.bert(bert_input, keep_batch=True)
                    bert_loss += l
                bert_loss /= self.n_mask
            # if self.use_my_ratio:
            #     if self.sequencelevel:
            #         if loss_reward.shape[1] == bert_ratio.shape[1]:
            #             loss_reward = loss_reward * bert_ratio.detach() # 先将[1024, 5]与[1024, 5]相乘
            #         elif loss_reward.shape[1] < bert_ratio.shape[1]:
            #             loss_reward *= bert_ratio[:, 0:loss_reward.shape[1]].detach() # 先将[1024, 5]与[1024, 5]相乘
            #         else:
            #             loss_reward[:, 0:bert_ratio.shape[1]] *= bert_ratio.detach() # 先将[1024, 5]与[1024, 5]相乘
            #         loss_reward = (loss_reward * mask).sum(dim=-1) / mask.sum(dim=-1) # 然后取遮罩部分的预测错误作为奖励值
            #     else:
            #         loss_reward[:, -1] *= bert_ratio.detach() # 先将[1024, 5]与[1024, 5]相乘
            #         loss_reward = (loss_reward * mask).sum(dim=-1) / mask.sum(dim=-1) # 然后取遮罩部分的预测错误作为奖励值
            # bert_loss
            self.bert_opt.zero_grad(set_to_none=True)
            (bert_loss.mean()).backward()
            self.bert_opt.step()

            if self.use_my_ratio:
                if self.sequencelevel:
                    bert_ratio = torch.mean(bert_ratio, dim=1, keepdim=True)
                bert_ratio = bert_ratio.view(*self.rewards.shape).detach()
            else:
                bert_ratio = 1

            # relabel rewards
            # if self.use_my_ratio:
            #     self.expl_r = loss_reward.detach().view(M, N, 1)
            # else:
            #     self.expl_r = bert_loss.detach().view(M, N, 1) # bert_loss是[16384],再把其恢复为self.expl_r:[32, 512, 1]


            # if self.use_my_ratio:
            self.expl_r = bert_loss.detach().view(M, N, 1) * bert_ratio
            # else:
            #     self.expl_r = bert_loss.detach().view(M, N, 1) # bert_loss是[16384],再把其恢复为self.expl_r:[32, 512, 1]

        self.rewards += self.expl_r * self.k_expl * anneal_weight  # self.rewards形状为[32, 512, 1]

        return self.expl_r.mean()

    #TODO：
    def get_expl_ratio(self):
        M, N, T, D = self.seq_obs_ratio.shape
        seq_observation = self.seq_obs_ratio.view(M * N, T, D) # self.seq_obs:[32, 512, 5, 384], bert_input:[16384, 5, 384]
        seq_next_observation = self.seq_next_obs_ratio.view(M * N, T, D) # self.seq_obs:[32, 512, 5, 384], bert_input:[16384, 5, 384]
        M, N, T, D = self.seq_act_ratio.shape
        seq_next_act = self.seq_act_ratio.view(M * N, T, D) # self.seq_obs:[32, 512, 5, 23], bert_input:[16384, 5, 23]
        bert_loss, summed_attn = self.bert_ratio(seq_observation.detach(), seq_next_act.detach(), seq_next_observation.detach(), keep_batch=True)
        # optimize BERT
        self.bert_ratio_opt.zero_grad(set_to_none=True)
        (bert_loss.mean()).backward()
        self.bert_ratio_opt.step()

        # return bert loss (N,) as exploration reward
        return summed_attn

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones.cpu() # [32, 512, 1]
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1) # [16384, 1]
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0])
        )
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        if self.sampler == "sequential":
            # For physics-based RL, each environment is already randomized. There is no value to doing random sampling
            # but a lot of CPU overhead during the PPO process. So, we can just switch to a sequential sampler instead
            subset = SequentialSampler(range(batch_size))
        elif self.sampler == "random":
            subset = SubsetRandomSampler(range(batch_size))

        batch = BatchSampler(subset, mini_batch_size, drop_last=True)
        return batch
