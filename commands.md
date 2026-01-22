### collect data

```bash
python scripts/libero_100_collect_demonstrations2.py --bddl-file libero/libero/bddl_files/custom/MAZE_SCENE_push_the_ball_to_the_goal.bddl --num-demonstration 50 --device spacemouse
```

### recover hdf5
```bash
python scripts/recover_demo_hdf5.py --directory /home/ubuntu/users/wyg/collected_data/move_the_block_marked_with_the_number_1_to_the_green_area_according_to_the_rules_of_the_number_sliding_puzzle/ --out-dir /home/ubuntu/users/wyg/collected_data/move_the_block_marked_with_the_number_1_to_the_green_area_according_to_the_rules_of_the_number_sliding_puzzle/demo --bddl-file libero/libero/bddl_files/custom_new/huarongdao.bddl
```

### generate init file
```bash
python scripts/generate_init_states.py \
  --bddl-folder libero/libero/bddl_files/custom \
  --save-folder libero/libero/init_files/custom \
  --num-states 20
```

### replay and convert

bridge converted
```bash
python scripts/create_dataset.py --demo-file /home/ubuntu/users/wyg/collected_data/pick_up_the_brick_and_place_it_across_the_gap/demo/demo.hdf5 --dataset-path /home/ubuntu/users/wyg/collected_data/ --dataset-name pick_up_the_brick_and_place_it_across_the_gap --use-camera-obs
```

maze converted
```bash
python scripts/create_dataset.py --demo-file /home/ubuntu/users/wyg/collected_data/push_the_ball_from_the_green_region_to_the_red_region_in_the_maze_on_the_table_without_picking_it_up/demo/demo.hdf5 --dataset-path /home/ubuntu/users/wyg/collected_data/ --dataset-name push_the_ball_from_the_green_region_to_the_red_region_in_the_maze_on_the_table_without_picking_it_up --use-camera-obs
```

hanoi converted
```bash
python scripts/create_dataset.py --demo-file /home/ubuntu/users/wyg/collected_data/move_all_the_rings_to_the_blue_stand_according_to_the_rules_of_the_tower_of_hanoi/demo/demo.hdf5 --dataset-path /home/ubuntu/users/wyg/collected_data/ --dataset-name move_all_the_rings_to_the_blue_stand_according_to_the_rules_of_the_tower_of_hanoi --use-camera-obs
```

huarongdao converted
```bash
python scripts/create_dataset.py --demo-file /home/ubuntu/users/wyg/collected_data/move_the_block_marked_with_the_number_1_to_the_green_area_according_to_the_rules_of_the_number_sliding_puzzle/demo/demo.hdf5 --dataset-path /home/ubuntu/users/wyg/collected_data/ --dataset-name move_the_block_marked_with_the_number_1_to_the_green_area_according_to_the_rules_of_the_number_sliding_puzzle --use-camera-obs
```

seesaw converted
```bash
python scripts/create_dataset.py --demo-file /home/ubuntu/users/wyg/collected_data/pick_up_the_two_cyan_blocks_and_place_them_on_the_ends_of_the_seesaw_to_make_the_seesaw_low_on_the_left_and_high_on_the_right/demo/demo.hdf5 --dataset-path /home/ubuntu/users/wyg/collected_data/ --dataset-name pick_up_the_two_cyan_blocks_and_place_them_on_the_ends_of_the_seesaw_to_make_the_seesaw_low_on_the_left_and_high_on_the_right --use-camera-obs
```

bridge3 converted
```bash
python scripts/create_dataset.py --demo-file /home/ubuntu/users/wyg/collected_data/place_the_left_brick_on_the_left_platform,_the_right_brick_on_the_right_platform,_and_the_middle_brick_across_the_gap_on_top_of_the_first_two_bricks/demo/demo.hdf5 --dataset-path /home/ubuntu/users/wyg/collected_data/ --dataset-name place_the_left_brick_on_the_left_platform,_the_right_brick_on_the_right_platform,_and_the_middle_brick_across_the_gap_on_top_of_the_first_two_bricks --use-camera-obs
```

shudu converted
```bash
python scripts/create_dataset.py --demo-file /home/ubuntu/users/wyg/collected_data/place_the_number_blocks_in_the_correct_positions_to_ensure_that_there_are_no_repeating_numbers_in_any_row_or_column/demo/demo.hdf5 --dataset-path /home/ubuntu/users/wyg/collected_data/ --dataset-name place_the_number_blocks_in_the_correct_positions_to_ensure_that_there_are_no_repeating_numbers_in_any_row_or_column --use-camera-obs
```

tictactoe1 converted
```bash
python scripts/create_dataset.py --demo-file /home/ubuntu/users/wyg/collected_data/tic-tac-toe_make_block_x_a_line/demo/demo.hdf5 --dataset-path /home/ubuntu/users/wyg/collected_data/ --dataset-name tic-tac-toe_make_block_x_a_line --use-camera-obs
```

tictactoe2 converted
```bash
python scripts/create_dataset.py --demo-file /home/ubuntu/users/wyg/collected_data/tic-tac-toe_prevent_block_o_from_forming_a_line/demo/demo.hdf5 --dataset-path /home/ubuntu/users/wyg/collected_data/ --dataset-name tic-tac-toe_prevent_block_o_from_forming_a_line --use-camera-obs
```

huanfang converted
```bash
python scripts/create_dataset.py --demo-file /home/ubuntu/users/wyg/collected_data/place_the_number_tiles_in_the_correct_positions_so_that_the_sum_of_the_numbers_in_each_row_and_column_and_diagonal_is_equal/demo/demo.hdf5 --dataset-path /home/ubuntu/users/wyg/collected_data/ --dataset-name place_the_number_tiles_in_the_correct_positions_so_that_the_sum_of_the_numbers_in_each_row_and_column_and_diagonal_is_equal --use-camera-obs
```
