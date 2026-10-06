"""MetisWAM4D variant for the RoboDojo Memory / Open tracks, initialised from InternW0-Delta.

Video / Track uvd / Action experts as in ``metiswam4d``, with sparse episode memory in the Video expert
(episode frame 0 + the first frames of the most recent chunk windows), InternW0-Delta's 14-D absolute joint
actions and its 384x256 three-camera canvas.  ``metiswam4d`` itself is imported, never modified.
"""
