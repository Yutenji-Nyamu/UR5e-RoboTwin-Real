#!/usr/bin/env python
"""Thin wrapper around Zero-WAM's RoboTwin evaluation client that adds demo-condition switches.

Run from ``$ROBOTWIN_ROOT`` with ``PYTHONPATH=$ZERO_WAM_ROOT:$ROBOTWIN_ROOT`` (same as their launch_client.sh).
Environment switches:
  ZEROWAM_NO_ICL=1        send ``use_icl=False`` on reset (no human-video prompt at all)
  ZEROWAM_ICL_MAP=<file>  alternative ``ROBOTWIN_ICL_HUMAN_VIDEOS`` map (e.g. other-task demos) - or pass
                          ``--icl_human_video_map`` as usual
All other arguments are forwarded to ``evaluation.robotwin.eval_policy_client_openpi``.
"""
import os
import sys

from evaluation.robotwin import eval_policy_client_openpi as C
from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy

if os.environ.get("ZEROWAM_NO_ICL") == "1":
    _orig_infer = WebsocketClientPolicy.infer

    def _infer(self, obs):
        if isinstance(obs, dict) and obs.get("reset"):
            obs = dict(obs, use_icl=False, icl_video_path="", icl_latent_path="")
        return _orig_infer(self, obs)

    WebsocketClientPolicy.infer = _infer
    print("[probe] ICL disabled (use_icl=False on reset)", flush=True)

if __name__ == "__main__":
    C.Sapien_TEST()
    usr_args = C.parse_args_and_config()
    C.main(usr_args)
