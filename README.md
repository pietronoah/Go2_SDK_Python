# unitree_sdk2_python
Python interface for unitree sdk2

# Installation
Create a python venv using Python 3.10:
```bash
# Create and activate a virtual environment
python3.10 -m venv go2_sdk
source go2_sdk/bin/activate
```
Execute the following commands in the terminal:
```bash
git clone https://github.com/pietronoah/Go2_SDK_Python.git
cd Go2_SDK_Python
pip3 install -e .
```

## Run the example (trot / pronk)

Before running, determine the network interface name that connects to your robot (this is required as the `interface_code` argument in the example). You can list interfaces with:

```bash
ip a
```

Two policies are available, one for trot and one for pronk gait. Choose the desired one in the `config.yaml`:

Trot example:

```yaml
policy:
	gait: trot
```
Pronk example:

```yaml
policy:
	gait: pronk
```

Run the unified policy script:

```bash
# Run
python3 example/go2/low_level/policy_nn.py interface_code
```

Controller buttons (Xbox controller mapping)
-----------------------------------------

The example uses an Xbox-style joystick mapping. The buttons used by the script are:

- Y (safety / kill): pressing Y will trigger the safety stop and exit the robot process.
- A (sit down): pressing A will command the robot to sit down (runs the sit-down sequence).
- B (stand up): pressing B will command the robot to stand up (starts the stand-up sequence).

Make sure your controller is connected and recognized by `pygame` before running the example.


