"""Auto-generated ICL demo map (probe)."""
from pathlib import Path
_ROOT = Path("/m2v_intern_v3/danglingwei/codes/Zero-WAM_260923") / "data" / "HumanGen" / "human_data" / "robotwin" / "run_robotwin_20260728_013720_robotwin_n1000" / "samples"
def _video(s):
    return str(_ROOT / s / "generated_video_kling-v3.mp4")
ROBOTWIN_ICL_HUMAN_VIDEOS = {
    "place_object_scale": [_video("074_robotwin_Use_appropriate_arm_to_move_the_stapler_to_a_colored_mat")],  # demo of move_stapler_pad
    "stamp_seal": [_video("519_robotwin_There_are_three_blocks_on_the_table_the_color_of_the_blocks")],  # demo of stack_blocks_three
    "open_microwave": [_video("007_robotwin_If_there_is_one_bread_on_the_table_use_one_arm_to_grab_the")],  # demo of place_bread_basket
    "move_stapler_pad": [_video("273_robotwin_Use_an_arm_to_place_the_empty_cup_on_the_coaster")],  # demo of place_empty_cup
    "stack_blocks_three": [_video("086_robotwin_Use_one_arm_to_grab_the_object_and_put_it_on_the_scale")],  # demo of place_object_scale
    "place_bread_basket": [_video("206_robotwin_Grab_the_stamp_and_stamp_onto_the_specific_color_mat")],  # demo of stamp_seal
    "place_empty_cup": [_video("242_robotwin_Use_one_arm_to_open_the_microwave")],  # demo of open_microwave
}
