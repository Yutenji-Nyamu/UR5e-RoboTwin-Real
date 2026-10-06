"""Privileged scripted experts for RoboDojo (ARX X5 dual arm, Isaac Sim 5.1).

The expert runs inside the official Isaac client (``expert_client``): it reads ground-truth object poses / bounding
boxes from the scene, plans with RoboDojo's own cuRobo planner and emits 14-D absolute joint actions through the same
``take_action`` / ``get_obs`` cadence as a policy, so the teacher recorder (``iw0_deploy.Recorder4D``) writes the
episodes in the teacher schema.  One module per task (``general_pickup``, ``stack_blocks_by_language``) with an
``episode(scene)`` generator of joint commands; ``skills`` holds the shared scene / planning helpers.
"""
