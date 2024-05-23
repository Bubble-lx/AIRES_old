import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch as th
from torch import Tensor
from torch.nn import init

def normalize_rewards(norm_type, rewards, mean, std, eps=1e-5):
    """
    Normalize the input rewards using a specified normalization method (norm_type).
    [0] No normalization
    [1] Standardization per mini-batch
    [2] Standardization per rollout buffer
    [3] Standardization without subtracting the average reward
    """
    if norm_type <= 0:
        return rewards

    if norm_type == 1:
        # Standardization
        return (rewards - mean) / (std + eps)

    if norm_type == 2:
        # Min-max normalization
        min_int_rew = np.min(rewards)
        max_int_rew = np.max(rewards)
        mean_int_rew = (max_int_rew + min_int_rew) / 2
        return (rewards - mean_int_rew) / (max_int_rew - min_int_rew + eps)

    if norm_type == 3:
        # Standardization without subtracting the mean
        return rewards / (std + eps)
# (model_cnn_extractor): BatchNormCnnFeaturesExtractor(
#       (activation_fn): ReLU()
#       (cnn): Sequential(
#         (0): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (1): Conv2d(3, 32, kernel_size=(2, 2), stride=(1, 1))
#         (2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (3): ReLU()
#         (4): Conv2d(32, 64, kernel_size=(2, 2), stride=(1, 1))
#         (5): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (6): ReLU()
#         (7): Conv2d(64, 64, kernel_size=(2, 2), stride=(1, 1))
#         (8): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (9): ReLU()
#         (10): Flatten(start_dim=1, end_dim=-1)
#       )
#       (linear_layer): Sequential(
#         (0): Linear(in_features=1024, out_features=64, bias=True)
#         (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (2): ReLU()
#       )
#     )

# class InverseModelOutputHeads(nn.Module):
#     def __init__(
#         self,
#         features_dim: int,
#         latents_dim: int = 128,
#         activation_fn: Type[nn.Module] = nn.ReLU,
#         action_num: int = 0,
#         mlp_norm: NormType = NormType.NoNorm,
#         mlp_layers: int = 1,
#     ):
#         super(InverseModelOutputHeads, self).__init__()

#         modules = [
#             nn.Linear(features_dim * 2, latents_dim),curr_obs
#         for _ in range(1, mlp_layers):
#             modules += [
#                 nn.Linear(latents_dim, latents_dim),
#                 NormType.get_norm_layer_1d(mlp_norm, latents_dim),
#                 activation_fn(),
#             ]
#         modules.append(nn.Linear(latents_dim, action_num))
#         self.nn = nn.Sequential(*modules)

#     def forward(self, curr_emb: Tensor, next_emb: Tensor) -> Tensor:
#         inputs = th.cat([curr_emb, next_emb], dim=1)
#         return self.nn(inputs)

# class RNDOutputHeads(nn.Module):
#     def __init__(self,
#                  features_dim: int,
#                  latents_dim: int = 128,
#                  outputs_dim: int = 128,
#                  activation_fn: Type[nn.Module] = nn.ReLU,
#                  mlp_norm: NormType = NormType.NoNorm,
#                  mlp_layers: int = 1,
#     ):
#         super().__init__()

#         self.target = nn.Sequential(
#             nn.Linear(features_dim, latents_dim),
#             NormType.get_norm_layer_1d(mlp_norm, latents_dim),
#             activation_fn(),

#             nn.Linear(latents_dim, latents_dim),
#             NormType.get_norm_layer_1d(mlp_norm, latents_dim),
#             activation_fn(),

#             nn.Linear(latents_dim, outputs_dim),
#             NormType.get_norm_layer_1d(mlp_norm, outputs_dim),
#         )

#         self.predictor = nn.Sequential(
#             nn.Linear(features_dim, latents_dim),
#             NormType.get_norm_layer_1d(mlp_norm, latents_dim),
#             activation_fn(),

#             nn.Linear(latents_dim, outputs_dim),
#             NormType.get_norm_layer_1d(mlp_norm, outputs_dim),
#         )

#         for param in self.target.parameters():
#             param.requires_grad = False

#     def forward(self, emb: Tensor) -> Tuple[Tensor, Tensor]:
#         with th.no_grad():
#             target_outputs = self.target(emb)
#         predicted_outputs = self.predictor(emb)
#         return target_outputs, predicted_outputs

# class NGUOutputHeads(nn.Module):
#     def __init__(
#         self,
#         features_dim: int,
#         latents_dim: int = 128,
#         activation_fn: Type[nn.Module] = nn.ReLU,
#         action_num: int = 0,
#         mlp_norm: NormType = NormType.NoNorm,
#         mlp_layers: int = 1,
#         use_rnd: int = 0,
#     ):
#         super(NGUOutputHeads, self).__init__()
#         self.use_rnd = use_rnd
#         if use_rnd:
#             self.ngu_rnd_model = RNDOutputHeads(
#                 features_dim, latents_dim, latents_dim, activation_fn,
#                 mlp_norm, mlp_layers
#             )
#         self.ngu_inverse_model = InverseModelOutputHeads(
#             features_dim, latents_dim, activation_fn, action_num,
#             mlp_norm, mlp_layers
#         )

#     def inverse_forward(self, curr_emb: Tensor, next_emb: Tensor) -> Tensor:
#         return self.ngu_inverse_model(curr_emb, next_emb)

#     def rnd_forward(self, curr_emb: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
#         if self.use_rnd:
#             return self.ngu_rnd_model(curr_emb)
#         return None, None

# class InverseModelOutputHeads(nn.Module):
#     def __init__(
#         self,
#         features_dim: int,
#         latents_dim: int = 128,
#         activation_fn: Type[nn.Module] = nn.ReLU,
#         action_num: int = 0,
#         mlp_norm: NormType = NormType.NoNorm,
#         mlp_layers: int = 1,
#     ):
#         super(InverseModelOutputHeads, self).__init__()

#         modules = [
#             nn.Linear(features_dim * 2, latents_dim),
#             NormType.get_norm_layer_1d(mlp_norm, latents_dim),
#             activation_fn(),
#         ]
#         for _ in range(1, mlp_layers):
#             modules += [
#                 nn.Linear(latents_dim, latents_dim),
#                 NormType.get_norm_layer_1d(mlp_norm, latents_dim),
#                 activation_fn(),
#             ]
#         modules.append(nn.Linear(latents_dim, action_num))
#         self.nn = nn.Sequential(*modules)

#     def forward(self, curr_emb: Tensor, next_emb: Tensor) -> Tensor:
#         inputs = th.cat([curr_emb, next_emb], dim=1)
#         return self.nn(inputs)


# class BatchNormCnnFeaturesExtractor(nn.Module):
#     def __init__(self):
#         super(BatchNormCnnFeaturesExtractor, self).__init__()
        
#         # 定义CNN部分
#         self.cnn = nn.Sequential(
#             nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.Conv2d(3, 32, kernel_size=(2, 2), stride=(1, 1)),
#             nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=(2, 2), stride=(1, 1)),
#             nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=(2, 2), stride=(1, 1)),
#             nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(),
#             nn.Flatten(start_dim=1, end_dim=-1)  # 注意此处Flatten后面可能需要调整以匹配后续层的in_features
#         )
        
#         # 定义接在CNN后的全连接层
#         self.linear_layer = nn.Sequential(
#             nn.Linear(in_features=1024, out_features=64, bias=True),  # in_features需要根据实际输入大小调整
#             nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU()
#         )
    
#     def forward(self, x):
#         x = self.cnn(x)
#         x = self.linear_layer(x)
#         return x

# ----------------------------------------------------------------
class RunningMeanStd(object):
    """
    Implemented based on:
    - https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    - https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/mpi_util.py#L179-L214
    - https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/running_mean_std.py
    """
    def __init__(self, epsilon=1e-4, momentum=None, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        self.eps = epsilon
        self.momentum = momentum

    def clear(self):
        self.__init__(self.eps, self.momentum)

    @staticmethod
    def update_ema(old_data, new_data, momentum):
        if old_data is None:
            return new_data
        return old_data * momentum + new_data * (1.0 - momentum)

    def update(self, x):
        batch_mean, batch_std, batch_count = np.mean(x, axis=0), np.std(x, axis=0), x.shape[0]
        batch_var = np.square(batch_std)
        if self.momentum is None or self.momentum < 0:
            self.update_from_moments(batch_mean, batch_var, batch_count)
        else:
            self.mean = self.update_ema(self.mean, batch_mean, self.momentum)
            new_var = np.mean(np.square(x - self.mean))
            self.var = self.update_ema(self.var, new_var, self.momentum)
            self.std = np.sqrt(self.var)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count
        self.mean = new_mean
        self.var = new_var
        self.std = np.sqrt(new_var)
        self.count = new_count

class RNDOutputHeads(nn.Module):
    def __init__(self, obs_dim, feature_dim=288, hidden_dim=256, lr=1e-3):
        super().__init__()

        # self.target = nn.Sequential(
        #     nn.Linear(in_features=64, out_features=64, bias=True),
        #     # nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=64, out_features=64, bias=True),
        #     # nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=64, out_features=64, bias=True),
        #     # nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # )

        # self.predictor = nn.Sequential(
        #     nn.Linear(in_features=64, out_features=64, bias=True),
        #     # nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=64, out_features=64, bias=True),
        #     # nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # )

        self.target = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        # freeze target network
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x):
        predictor_output = self.predictor(x)
        with torch.no_grad():
            target_output = self.target(x)
        return target_output, predictor_output


class NGU(nn.Module):
    def __init__(self, obs_dim, action_dim, feature_dim=288, hidden_dim=256,
        lr=1e-3):
        super().__init__()
        self.ngu_dst_momentum = 0.997
        self.ngu_use_rnd = True
        self.rnd_err_norm = 1
        self.rnd_err_momentum = -1
        self.ngu_knn_k = 10
        self.ngu_moving_avg_dists = RunningMeanStd(momentum=self.ngu_dst_momentum)
        self.rnd_err_running_stats = RunningMeanStd(momentum=self.rnd_err_momentum)

        # s => phi(s)
        self.enc = nn.Linear(obs_dim, feature_dim)

        self.inv_model_deir = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            # nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),# 感觉像是可有可无
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.rnd_forward = RNDOutputHeads(obs_dim)

        self.inv_criterion = nn.MSELoss()
        self.for_criterion = nn.MSELoss(reduction='none')
        self.inv_opt = torch.optim.Adam(
            list(self.enc.parameters()) + list(self.inv_model_deir.parameters()),
            lr=lr)
        self.for_opt = torch.optim.Adam(self.rnd_forward.predictor.parameters(), lr=lr)

    def forward_deir(self, curr_obs, next_obs, last_mems, curr_act, curr_dones, i):
        # curr_embs = None
        # next_embs = None
        curr_mems = None

        obs_feat = self.enc(curr_obs) # obs_feat:[512, 288] # curr_obs:[512, 384]
        next_obs_feat = self.enc(next_obs)

        curr_embs = obs_feat.detach().clone()
        next_embs = next_obs_feat.detach().clone()

        pred_act = self.inv_model_deir(torch.cat([obs_feat, next_obs_feat], dim=-1))

        curr_dones = curr_dones.view(-1)
        n_samples = (1 - curr_dones).sum()
        inv_loss = self.inv_criterion(pred_act, curr_act) # inv_loss:[0.3364]
        if i % 7 == 0:
            self.inv_opt.zero_grad(set_to_none=True)
            (inv_loss.mean()).backward()
            self.inv_opt.step()

        # RND
        rnd_losses, rnd_loss, _ = None, None, None
        # if self.ngu_use_rnd:
        curr_obs_detached = curr_obs.detach().clone()
        tgt_out, prd_out = self.rnd_forward(curr_obs_detached)
        rnd_losses = self.for_criterion(prd_out, tgt_out) # rnd_loss:[512, 288]
        if i % 7 == 0:
            self.for_opt.zero_grad(set_to_none=True)
            (rnd_losses.mean()).backward()
            self.for_opt.step()

        rnd_losses = rnd_losses.detach().mean(dim=-1) # rnd_loss：[512]
        rnd_loss = rnd_losses.sum() * (1 / n_samples if n_samples > 0 else 0.0) # rnd_loss:tensor

        # ngu_loss = inv_loss
        # if self.ngu_use_rnd:
        #     ngu_loss = ngu_loss + rnd_loss

        return inv_loss.detach(), curr_embs, next_embs, rnd_losses.detach(), rnd_loss.detach(), curr_mems

    def forward(self, obs, next_obs, actions, obs_history, dones):

        # obs.shape[0] # 512
        n_steps = obs.shape[0] # 32
        batch_size = obs.shape[1]
        reward_all = []
        for i in range(n_steps):
            for env_id in range(batch_size):
                if dones[i, env_id, :]:
                    obs_history[env_id] = None

            last_mems = None
            # 需要的参数： curr_dones、obs_history
            curr_dones = dones[i, :]

            inv_loss, ngu_curr_embs, ngu_next_embs, ngu_rnd_losses, ngu_rnd_loss, _ = \
                    self.forward_deir(obs[i,:,:], next_obs[i,:,:], last_mems, actions[i,:,:], curr_dones, i)

            if ngu_rnd_losses is not None:
                ngu_rnd_error = ngu_rnd_losses.clone().cpu().numpy()

                if self.rnd_err_norm > 0:
                    # Normalize RND error per step
                    self.rnd_err_running_stats.update(ngu_rnd_error)
                    ngu_rnd_error = normalize_rewards(
                        norm_type=self.rnd_err_norm,
                        rewards=ngu_rnd_error,
                        mean=self.rnd_err_running_stats.mean,
                        std=self.rnd_err_running_stats.std,
                    )

                ngu_lifelong_rewards = ngu_rnd_error + 1

            # Create IRs
            int_rews = np.zeros(batch_size, dtype=np.float32)

            for env_id in range(batch_size):
                # Update historical observation embeddings
                curr_emb = ngu_curr_embs[env_id].view(1, -1)
                next_emb = ngu_next_embs[env_id].view(1, -1)
                obs_embs = obs_history[env_id]
                new_embs = [curr_emb, next_emb] if obs_embs is None else [obs_embs, next_emb]
                obs_embs = th.cat(new_embs, dim=0)
                obs_history[env_id] = obs_embs

                # Implemented based on the paper of NGU (Algorithm 1)
                episodic_reward = 0.0
                if obs_embs.shape[0] > 1:
                    # Compute the k-nearest neighbours of f (x_t) in M and store them in a list N_k
                    # - d is the Euclidean distance and
                    # - d^2_m is a running average of the squared Euclidean distance of the k-nearest neighbors.
                    knn_dists = self.calc_euclidean_dists(obs_embs[:-1], obs_embs[-1]) ** 2
                    knn_dists = knn_dists.clone().cpu().numpy()
                    knn_dists = np.sort(knn_dists)[:self.ngu_knn_k]
                    # Update the moving average d^2_m with the list of distances d_k
                    self.ngu_moving_avg_dists.update(knn_dists)
                    moving_avg_dist = self.ngu_moving_avg_dists.mean
                    # Normalize the distances d_k with the updated moving average d^2_m
                    normalized_dists = knn_dists / (moving_avg_dist + 1e-5)
                    # Cluster the normalized distances d_n
                    # i.e. they become 0 if too small and 0k is a list of k zeros
                    normalized_dists = np.maximum(normalized_dists - 0.008, np.zeros_like(knn_dists))
                    # Compute the Kernel values between the embedding f (x_t) and its neighbours N_k
                    kernel_values = 0.0001 / (normalized_dists + 0.0001)
                    # Compute the similarity between the embedding f (x_t) and its neighbours N_k
                    simlarity = np.sqrt(kernel_values.sum()) + 0.001
                    # Compute the episodic intrinsic reward at time t
                    if simlarity <= 8:
                        episodic_reward += 1 / simlarity

                if self.ngu_use_rnd and ngu_lifelong_rewards is not None:
                    L = 5.0  # L is a chosen maximum reward scaling (default: 5)
                    lifelong_reward = min(max(ngu_lifelong_rewards[env_id], 1.0), L)
                    int_rews[env_id] += episodic_reward * lifelong_reward
                else:
                    int_rews[env_id] += episodic_reward
            reward_all.append(int_rews)

        # reward_all由，[32，512]变成[16384]
        reward_matrix = np.stack(reward_all)
        # 转置数组
        reward_matrix = reward_matrix.T
        reward_flattened = reward_matrix.flatten()
        # reward_flattened = normalize_rewards(
        #     norm_type=1,
        #     rewards=reward_flattened,
        #     mean=reward_flattened.mean(),
        #     std=reward_flattened.std(),
        #     eps=1e-5,
        # )

        # Rescale by IR coef
        reward_flattened *= 0.1

        return reward_flattened


    @staticmethod
    @th.jit.script
    def calc_euclidean_dists(x : Tensor, y : Tensor):
        """
        Calculate the Euclidean distances between two batches of embeddings.
        Input shape: [n, d]
        Return: ((x - y) ** 2).sum(dim=-1) ** 0.5
        """
        features_dim = x.shape[-1]
        x = x.view(1, -1, features_dim)
        y = y.view(1, -1, features_dim)
        return th.cdist(x, y, p = 2.0)[0]



        # update inverse dynamics model and get features
        obs_feat, next_obs_feat = self.inverse_dynamics(obs, next_obs, actions)
        # predict next_obs using forward dynamics model and update model
        icm_loss = self.forward_dynamics(obs_feat, next_obs_feat, actions)
        icm_loss = icm_loss.detach().mean(dim=-1)
        return icm_loss

    # def forward(self, obs, next_obs, actions):
    #     # update inverse dynamics model and get features
    #     obs_feat, next_obs_feat = self.inverse_dynamics(obs, next_obs, actions)
    #     # predict next_obs using forward dynamics model and update model
    #     icm_loss = self.forward_dynamics(obs_feat, next_obs_feat, actions)
    #     icm_loss = icm_loss.detach().mean(dim=-1)
    #     return icm_loss
