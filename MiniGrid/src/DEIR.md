## DEIR

---

### 运行指令

```bash
DEIR：
python train.py --int_rew_source=DEIR --env_source=minigrid --game_name=DoorKey-16x16
python train.py --int_rew_source=DEIR --env_source=minigrid --game_name=DoorKey-16x16  --use_my_ratio=True

ICM：
python train.py --int_rew_source=ICM --env_source=minigrid --game_name=DoorKey-16x16  --total_steps=2e6
python train.py --int_rew_source=ICM --env_source=minigrid --game_name=DoorKey-16x16 --use_my_ratio=True  --total_steps=2e6

RND：
python train.py --int_rew_source=RND --env_source=minigrid --game_name=DoorKey-16x16 --model_features_dim=64 --model_latents_dim=64
python train.py --int_rew_source=RND --env_source=minigrid --game_name=DoorKey-16x16 --model_features_dim=64 --model_latents_dim=64 --use_my_ratio=True

NGU：
python train.py --int_rew_source=NGU --env_source=minigrid --game_name=DoorKey-16x16 --model_features_dim=64 --model_latents_dim=64
python train.py --int_rew_source=NGU --env_source=minigrid --game_name=DoorKey-16x16 --model_features_dim=64 --model_latents_dim=64  --use_my_ratio=True

NovelD：
python train.py --int_rew_source=NovelD --env_source=minigrid --game_name=DoorKey-16x16 --model_features_dim=64 --model_latents_dim=64
python train.py --int_rew_source=NovelD --env_source=minigrid --game_name=DoorKey-16x16 --model_features_dim=64 --model_latents_dim=64  --use_my_ratio=True

MIMEx：
python train.py --int_rew_source=MIMEx --env_source=minigrid --game_name=DoorKey-16x16 --model_features_dim=64 --model_latents_dim=64
python train.py --int_rew_source=MIMEx --env_source=minigrid --game_name=DoorKey-16x16 --model_features_dim=64 --model_latents_dim=64  --use_my_ratio=True
```



测试指令：

```bash
python train.py --int_rew_source=ICM  --env_source=minigrid --game_name=KeyCorridorS6R3  --total_steps=6000000
python train.py --int_rew_source=ICM  --env_source=minigrid --game_name=KeyCorridorS6R3  --total_steps=6000000  --use_my_ratio=True

python train.py --int_rew_source=ICM  --env_source=minigrid --game_name=KeyCorridorS4R3  --total_steps=6000000
python train.py --int_rew_source=ICM  --env_source=minigrid --game_name=KeyCorridorS4R3  --total_steps=6000000  --use_my_ratio=True
python train.py --int_rew_source=DEIR  --env_source=minigrid --game_name=KeyCorridorS4R3  --total_steps=6000000
```

四种场景

```bash
Standard：
python src/train.py --int_rew_source=DEIR --env_source=minigrid  --game_name=MultiRoom-N6 --model_features_dim=64 --model_latents_dim=64 --total_steps=2000000 --use_my_ratio=false
Reduced view sizes：
python src/train.py --int_rew_source=DEIR --env_source=minigrid  --game_name=MultiRoom-N6 --model_features_dim=64 --model_latents_dim=64 --total_steps=2000000 --use_my_ratio=false
Noisy observations：
python src/train.py --int_rew_source=DEIR --env_source=minigrid  --game_name=MultiRoom-N6 --model_features_dim=64 --model_latents_dim=64 --total_steps=2000000 --use_my_ratio=false
Invisible obstacles：
python src/train.py --int_rew_source=DEIR --env_source=minigrid  --game_name=MultiRoom-N6 --model_features_dim=64 --model_latents_dim=64 --total_steps=2000000 --use_my_ratio=false
```

PPOsetting

```bash
Minigrid：
OMFull (∗)：
--n_steps=256 --num_processes=64 --n_epochs=3 --model_n_epochs=3 --batch_size=2048 --ent_coef=5e-4 --adv_norm=0 --learning_rate=1e-4 --ext_rew_coef=10
ProcGen：
--n_steps=256 --num_processes=64 --n_epochs=3 --model_n_epochs=3 --batch_size=2048  --adv_norm=0 --learning_rate=1e-4

DEIR:
MiniGrid :
OMFull (∗): --int_rew_coef=1e-3
ProcGen: --int_rew_coef=5e-2
备注：ProcGen的Caveflyer：--int_rew_coef=5e-3

NovelD:
MiniGrid :--int_rew_coef=3e-2 --rnd_err_norm=0
OMFull (∗):--int_rew_coef=3e-3 --rnd_err_norm=0
ProcGen:--int_rew_coef=3e-2 --rnd_err_norm=0
备注：


NGU:
MiniGrid : --int_rew_coef=1e-3 --int_rew_momentum=0
OMFull (∗):
ProcGen: --int_rew_coef=3e-4 --int_rew_momentum=0
备注：


RND:
MiniGrid : --int_rew_coef=3e-3 --rnd_err_norm=0
OMFull (∗):
ProcGen: --rnd_err_norm=0
备注：

ICM:
MiniGrid : --int_rew_coef=1e-2
OMFull (∗):
ProcGen: --int_rew_coef=1e-4
备注：

```

MultiRoom-N6：

```bash
python train.py  --env_source=minigrid  --game_name=MultiRoom-N6 --total_steps=2000000   --int_rew_source=DEIR

python train.py  --env_source=minigrid  --game_name=MultiRoom-N6 --total_steps=2000000  --int_rew_source=NovelD --int_rew_coef=3e-2 --rnd_err_norm=0 --model_features_dim=64 --model_latents_dim=64

python train.py  --env_source=minigrid  --game_name=MultiRoom-N6 --total_steps=2000000 --model_features_dim=64 --model_latents_dim=64 --int_rew_source=NGU --int_rew_coef=1e-3 --int_rew_momentum=0

python train.py  --env_source=minigrid  --game_name=MultiRoom-N6 --total_steps=2000000 --model_features_dim=64 --model_latents_dim=64 --int_rew_source=RND  --int_rew_coef=3e-3 --rnd_err_norm=0

python train.py  --env_source=minigrid  --game_name=MultiRoom-N6 --total_steps=2000000 --model_features_dim=64 --model_latents_dim=64 --int_rew_source=ICM  --int_rew_coef=1e-2
```



conda activate deir

python train.py --game_name=MultiRoom-N6 --total_steps=2000000 --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --use_my_ratio=true --attn_ratio_weight=8.0



conda activate deir

python train.py --game_name=MultiRoom-N6 --total_steps=2000000 --int_rew_source=MIMEx







python train.py --game_name=MultiRoom-N6 --total_steps=2000000 --model_features_dim=64 --model_latents_dim=64 --int_rew_source=MIMEx --use_my_ratio=true



DEIR方法：

python train.py --int_rew_source=DEIR --env_source=minigrid --game_name=MultiRoom-N6  --total_steps=2000000 --image_noise_scale=0.1

NGU方法：

python train.py --model_features_dim=64 --model_latents_dim=64 --int_rew_source=NGU --env_source=minigrid --game_name=MultiRoom-N6 --total_steps=2000000 --image_noise_scale=0.1

NGU+我的方法

python train.py --model_features_dim=64 --model_latents_dim=64 --int_rew_source=NGU --env_source=minigrid --game_name=MultiRoom-N6 --use_my_ratio=True --total_steps=2000000 --attn_selet_way=1 --attn_ratio_weight=2.0 --image_noise_scale=0.1

---



DEIR+我的方法：

python train.py --int_rew_source=DEIR --env_source=minigrid --game_name=MultiRoom-N6  --total_steps=2000000 --image_noise_scale=0.1  --use_my_ratio=True --attn_selet_way=1 --attn_ratio_weight=2.0

想要测试的其他场景：DoorKey-16x16（有noise）   FourRooms  Reduced view sizes

---

FourRooms

python train.py --int_rew_source=DEIR --game_name=FourRooms  --total_steps=2000000

python train.py --int_rew_source=DEIR --game_name=FourRooms  --total_steps=2000000  --use_my_ratio=True

python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=FourRooms  --total_steps=2000000

python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=FourRooms  --total_steps=2000000  --use_my_ratio=True

python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=FourRooms  --total_steps=2000000

python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=FourRooms  --total_steps=2000000  --use_my_ratio=True

python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=FourRooms  --total_steps=2000000

python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=FourRooms  --total_steps=2000000  --use_my_ratio=True

Dynamic-Obstacles-5x5

python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=GoToDoor-5x5  --total_steps=2000000

```
MiniGrid-Dynamic-Obstacles-5x5-v0
MiniGrid-Dynamic-Obstacles-Random-5x5-v0
MiniGrid-Dynamic-Obstacles-6x6-v0
MiniGrid-Dynamic-Obstacles-Random-6x6-v0
MiniGrid-Dynamic-Obstacles-8x8-v0
MiniGrid-Dynamic-Obstacles-16x16-v0
python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=Dynamic-Obstacles-Random-6x6  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=Dynamic-Obstacles-Random-6x6  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=Dynamic-Obstacles-8x8  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=Dynamic-Obstacles-Random-6x6  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=Dynamic-Obstacles-Random-6x6  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=Dynamic-Obstacles-8x8  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true





MiniGrid-Fetch-5x5-N2-v0
MiniGrid-Fetch-6x6-N2-v0
MiniGrid-Fetch-8x8-N3-v0

没有测试
python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=Fetch-8x8-N3  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=Fetch-8x8-N3  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=Fetch-8x8-N3  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=Fetch-8x8-N3  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=Fetch-8x8-N3  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=Fetch-8x8-N3   --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true


MiniGrid-MemoryS17Random-v0
MiniGrid-MemoryS13Random-v0
MiniGrid-MemoryS13-v0
MiniGrid-MemoryS11-v0
python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=MemoryS11  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=MemoryS11  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=MemoryS11  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=MemoryS11  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=MemoryS11  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=MemoryS11   --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true

python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=GoToDoor-8x8  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=GoToDoor-8x8  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=GoToDoor-8x8  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=GoToDoor-8x8  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=GoToDoor-8x8  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=GoToDoor-8x8   --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true

MiniGrid-GoToDoor-5x5-v0
MiniGrid-GoToDoor-6x6-v0
MiniGrid-GoToDoor-8x8-v0
没有测试：
MiniGrid-LockedRoom-v0 ：感觉非常难
python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=LockedRoom  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=LockedRoom  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=LockedRoom  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=LockedRoom  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=LockedRoom  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=LockedRoom   --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
MiniGrid-RedBlueDoors-6x6-v0
MiniGrid-RedBlueDoors-8x8-v0
python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=RedBlueDoors-8x8  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=RedBlueDoors-8x8  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=RedBlueDoors-8x8  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=RedBlueDoors-8x8  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=RedBlueDoors-8x8  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=LocRedBlueDoors-8x8kedRoom   --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
MiniGrid-Unlock-v0

MiniGrid-DistShift2-v0
python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=DistShift2  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=DistShift2  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=DistShift2  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=DistShift2  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=DistShift2  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=DistShift2   --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
Lava :
MiniGrid-LavaCrossingS9N1-v0
MiniGrid-LavaCrossingS9N2-v0
MiniGrid-LavaCrossingS9N3-v0
MiniGrid-LavaCrossingS11N5-v0
python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=LavaCrossingS9N3  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=LavaCrossingS9N3  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=LavaCrossingS9N3  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=LavaCrossingS9N3  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=LavaCrossingS9N3  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=LavaCrossingS9N3   --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
otherwise :
MiniGrid-SimpleCrossingS9N1-v0
MiniGrid-SimpleCrossingS9N2-v0
MiniGrid-SimpleCrossingS9N3-v0
MiniGrid-SimpleCrossingS11N5-v0
python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=SimpleCrossingS9N3  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=SimpleCrossingS9N3  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=SimpleCrossingS9N3  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=SimpleCrossingS9N3  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=SimpleCrossingS9N3  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=SimpleCrossingS9N3   --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true

MiniGrid-BlockedUnlockPickup-v0
python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=BlockedUnlockPickup  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=BlockedUnlockPickup  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=BlockedUnlockPickup  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=BlockedUnlockPickup  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=BlockedUnlockPickup  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=BlockedUnlockPickup   --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true


执行测试的项目：
RedBlueDoors-8x8
DistShift2
BlockedUnlockPickup
SimpleCrossingS9N3
LavaCrossingS9N3

python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=RedBlueDoors-8x8  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=RedBlueDoors-8x8  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=RedBlueDoors-8x8  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=RedBlueDoors-8x8  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=RedBlueDoors-8x8  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=RedBlueDoors-8x8   --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true


python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=DistShift2  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=DistShift2  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=DistShift2  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=DistShift2  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=DistShift2  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=DistShift2   --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true


python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=BlockedUnlockPickup  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=BlockedUnlockPickup  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=BlockedUnlockPickup  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=BlockedUnlockPickup  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=BlockedUnlockPickup  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=BlockedUnlockPickup   --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true

python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=SimpleCrossingS9N3  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=SimpleCrossingS9N3  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=SimpleCrossingS9N3  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=SimpleCrossingS9N3  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=SimpleCrossingS9N3  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=SimpleCrossingS9N3   --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true

python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=LavaCrossingS9N3  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=LavaCrossingS9N3  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=LavaCrossingS9N3  --total_steps=2000000 --use_self_encoder=false
python train.py --int_rew_source=ICM  --model_features_dim=64 --model_latents_dim=64 --game_name=LavaCrossingS9N3  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=RND  --model_features_dim=64 --model_latents_dim=64 --game_name=LavaCrossingS9N3  --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true
python train.py --int_rew_source=NGU  --model_features_dim=64 --model_latents_dim=64 --game_name=LavaCrossingS9N3   --total_steps=2000000 --use_self_encoder=false --use_my_ratio=true

```



收集nijia数据：

python src/train.py --int_rew_source=DEIR --env_source=procgen  --game_name=ninja  --total_steps=80000000 --use_self_encoder=false --n_steps=256 --num_processes=64 --n_epochs=3 --model_n_epochs=3 --batch_size=2048  --adv_norm=0 --learning_rate=1e-4 --int_rew_coef=5e-2

python src/train.py --int_rew_source=DEIR --env_source=procgen  --game_name=ninja  --total_steps=80000000 --use_self_encoder=false --n_steps=256 --num_processes=64 --n_epochs=3 --model_n_epochs=3 --batch_size=512  --adv_norm=0 --learning_rate=1e-4 --int_rew_coef=5e-2

python src/train.py --int_rew_source=DEIR --env_source=procgen  --game_name=climber  --total_steps=80000000 --use_self_encoder=false --n_steps=256 --num_processes=64 --n_epochs=3 --model_n_epochs=3 --batch_size=512  --adv_norm=0 --learning_rate=1e-4 --int_rew_coef=5e-2

python src/train.py --int_rew_source=DEIR --env_source=procgen  --game_name=jumper  --total_steps=100000000 --use_self_encoder=false --n_steps=256 --num_processes=64 --n_epochs=3 --model_n_epochs=3 --batch_size=512  --adv_norm=0 --learning_rate=1e-4 --int_rew_coef=5e-2

python src/train.py --int_rew_source=DEIR --env_source=procgen  --game_name=caveflyer  --total_steps=100000000 --use_self_encoder=false --n_steps=256 --num_processes=64 --n_epochs=3 --model_n_epochs=3 --batch_size=512  --adv_norm=0 --learning_rate=1e-4 --int_rew_coef=5e-2
