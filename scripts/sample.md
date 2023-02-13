Stone:

python projector.py --name Stone_uncond --ckpt 800000.pt --starting_height_size 4 --mode sample --savename dddd --circular --channel_multiplier 1 --nocond_z --H_scale 10 --dataset Stone --no_cond

Metal:


python projector.py --name Metal_uncond --ckpt 800000.pt --starting_height_size 4 --mode sample --savename dddd --circular --channel_multiplier 1 --nocond_z --H_scale 5 --dataset Metal --no_cond

Tiles:

python projector.py --name Tiles --ckpt 800000.pt --starting_height_size 32 --mode sample --savename ddd --pat_path ./Data/Bricks_test_pat_demo --circular --channel_multiplier 1 --nocond_z --H_scale 10 --dataset Tile


Leather:


python projector.py --name Leather3_cir_stylereg --ckpt 520000.pt --starting_height_size 4 --mode sample --savename ddd --pat_path ./Data/Leather_test_pat_demo --circular --nocond_z --channel_multiplier 1 --H_scale 5 --dataset Leather