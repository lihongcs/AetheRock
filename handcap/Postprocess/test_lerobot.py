import sys
sys.path.insert(0, "./lerobot30_env")
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import inspect
print(inspect.getsource(LeRobotDataset.add_frame))
