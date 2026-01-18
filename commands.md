collect data:
```bash
python scripts/libero_100_collect_demonstrations2.py --bddl-file libero/libero/bddl_files/custom/MAZE_SCENE_push_the_ball_to_the_goal.bddl --num-demonstration 50 --device spacemouse
```

replay and convert
bridge
```bash
python scripts/create_dataset.py --demo-file /home/ubuntu/users/wyg/collected_data/pick_up_the_brick_and_place_it_across_the_gap/demo/demo.hdf5 --dataset-path /home/ubuntu/users/wyg/collected_data/ --dataset-name pick_up_the_brick_and_place_it_across_the_gap --use-camera-obs
```

maze
```bash
python scripts/create_dataset.py --demo-file /home/ubuntu/users/wyg/collected_data/push_the_ball_from_the_green_region_to_the_red_region_in_the_maze_on_the_table_without_picking_it_up/demo/demo.hdf5 --dataset-path /home/ubuntu/users/wyg/collected_data/ --dataset-name push_the_ball_from_the_green_region_to_the_red_region_in_the_maze_on_the_table_without_picking_it_up --use-camera-obs
```