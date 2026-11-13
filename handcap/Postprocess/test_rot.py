import numpy as np
from scipy.spatial.transform import Rotation as st
rot = st.from_rotvec(np.array([0.1, 0.2, 0.3]))
print("rot.as_matrix().shape:", rot.as_matrix().shape)
print("rot.as_matrix()[:, 0].shape:", rot.as_matrix()[:, 0].shape)
