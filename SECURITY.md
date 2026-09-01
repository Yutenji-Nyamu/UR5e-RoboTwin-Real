# Security and laboratory configuration

This public repository must not contain API keys, credentials, private datasets,
checkpoint files, or a filled laboratory configuration.

Copy `configs/lab.example.yaml` to the ignored `configs/lab.yaml` and store local
robot addresses, camera serial numbers, serial ports, and paths there. Report a
secret exposure privately to the repository owner; do not open a public issue
containing the value.

Robot motion is a physical safety boundary. Run motion commands only with a clear
workspace, correct PolyScope mode, conservative speed limits, and a person at the
emergency stop.
