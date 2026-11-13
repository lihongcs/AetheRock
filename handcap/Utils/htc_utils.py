import time
import ctypes
import numpy as np
import open3d as o3d
import xr
from ctypes import cast, byref, POINTER
from scipy.spatial.transform import Rotation as R

# Matrices defined in your original code
x_rotation_clockwise_90 = np.array([
    [1,  0,  0],
    [0,  0,  1],
    [0, -1,  0]
])

z_rotation_clockwise_90 = np.array([
    [ 0,  1,  0],
    [-1,  0,  0],
    [ 0,  0,  1]
])

y_rotation_clockwise_90 = np.array([
    [ 0,  0,  1],
    [ 0,  1,  0],
    [-1,  0,  0]
])

z_rotation_180 = np.array([
    [-1,  0, 0],
    [ 0, -1, 0],
    [ 0,  0, 1]
])

class HTCModule:
    def __init__(self):
        # 1. Define Coordinate Systems
        self.combined_world_rotation = np.dot(x_rotation_clockwise_90, z_rotation_clockwise_90)
        self.combined_world_rotation_inverse = self.combined_world_rotation.T
        
        # 2. Define Roles
        self.role_strings = [
            "handheld_object", "left_foot", "right_foot", "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow", "left_knee", "right_knee", "waist", "chest",
            "camera", "keyboard",
        ]

        # 3. Initialize OpenXR Context
        # We manually trigger the context manager methods to keep the session alive inside the class
        self.context = ContextObject(
            instance_create_info=xr.InstanceCreateInfo(
                enabled_extension_names=[
                    xr.MND_HEADLESS_EXTENSION_NAME,
                    xr.extension.HTCX_vive_tracker_interaction.NAME,
                ],
            ),
        )
        self.context.__enter__() # Manually enter context
        
        self.instance = self.context.instance
        self.session = self.context.session

        # 4. Setup Tracker Actions and Bindings
        self._setup_openxr_actions()

    def _setup_openxr_actions(self):
        """Internal helper to setup OpenXR paths and actions"""
        role_path_strings = [f"/user/vive_tracker_htcx/role/{role}" for role in self.role_strings]
        
        role_paths = (xr.Path * len(role_path_strings))(*[
            xr.string_to_path(self.instance, role_string) for role_string in role_path_strings
        ])

        pose_action = xr.create_action(
            action_set=self.context.default_action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name="tracker_pose",
                localized_action_name="Tracker Pose",
                count_subaction_paths=len(role_paths),
                subaction_paths=role_paths,
            ),
        )

        suggested_binding_paths = (xr.ActionSuggestedBinding * len(role_path_strings))(*[
            xr.ActionSuggestedBinding(
                pose_action, 
                xr.string_to_path(self.instance, f"{role_path_string}/input/grip/pose")
            ) for role_path_string in role_path_strings
        ])

        xr.suggest_interaction_profile_bindings(
            instance=self.instance, 
            suggested_bindings=xr.InteractionProfileSuggestedBinding(
                interaction_profile=xr.string_to_path(self.instance, "/interaction_profiles/htc/vive_tracker_htcx"),
                count_suggested_bindings=len(suggested_binding_paths), 
                suggested_bindings=suggested_binding_paths,
            )
        )

        self.tracker_action_spaces = (xr.Space * len(role_paths))(*[
            xr.create_action_space(
                session=self.session, 
                create_info=xr.ActionSpaceCreateInfo(action=pose_action, subaction_path=role_path)
            ) for role_path in role_paths
        ])

        # Get Enumerate Function Pointer
        self.enumerateViveTrackerPathsHTCX = cast(
            xr.get_instance_proc_addr(self.instance, "xrEnumerateViveTrackerPathsHTCX"),
            xr.PFN_xrEnumerateViveTrackerPathsHTCX
        )

    def _quaternion_to_rotation_matrix(self, q):
        """Helper: Quaternion [x, y, z, w] to Rotation Matrix"""
        x, y, z, w = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])

    def _rotation_matrix_to_quaternion(self, R):
        """Helper: Rotation Matrix to Quaternion [x, y, z, w]"""
        # Note: Scipy is more robust, but using the logic from your provided script class
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return np.array([x, y, z, w])

    def compute_rel_transform(self, position, orientation):
        """
        Applies the coordinate transformations defined in the original script.
        
        Args:
            position: [x, y, z]
            orientation: [x, y, z, w]
            
        Returns:
            position_transformed: np.array [x, y, z]
            orientation_transformed: np.array [x, y, z, w]
        """
        # 1. Transform Position (World Rotation Inverse)
        position_transformed = np.dot(self.combined_world_rotation_inverse, position)

        # 2. Transform Orientation
        # Convert quat to matrix
        object_rotation_matrix = self._quaternion_to_rotation_matrix(orientation)
        
        # Apply specific rotations found in your main loop
        transformed_rotation_matrix = object_rotation_matrix
        transformed_rotation_matrix = np.dot(transformed_rotation_matrix, y_rotation_clockwise_90)
        transformed_rotation_matrix = np.dot(transformed_rotation_matrix, z_rotation_180)
        
        # Convert back to quat
        orientation_transformed = self._rotation_matrix_to_quaternion(transformed_rotation_matrix)
        
        return position_transformed, orientation_transformed

    def world(self):
        """
        Generator that yields dictionary of detected trackers and their transformed poses.
        
        Yields:
            dict: { "role_name": {"pos": np.array, "quat": np.array}, ... }
        """
        for frame_state in self.context.frame_loop():
            # Handle Session State
            if self.context.session_state != xr.SessionState.FOCUSED:
                continue

            # Sync Actions
            active_action_set = xr.ActiveActionSet(
                action_set=self.context.default_action_set, 
                subaction_path=xr.NULL_PATH
            )
            xr.sync_actions(
                session=self.session, 
                sync_info=xr.ActionsSyncInfo(
                    count_active_action_sets=1, 
                    active_action_sets=ctypes.pointer(active_action_set)
                )
            )

            # Enumerate Trackers (Required by logic)
            n_paths = ctypes.c_uint32(0)
            self.enumerateViveTrackerPathsHTCX(self.instance, 0, byref(n_paths), None)
            vive_tracker_paths = (xr.ViveTrackerPathsHTCX * n_paths.value)(*([xr.ViveTrackerPathsHTCX()] * n_paths.value))
            self.enumerateViveTrackerPathsHTCX(self.instance, n_paths, byref(n_paths), vive_tracker_paths)

            frame_data = {}

            # Locate Spaces
            for index, space in enumerate(self.tracker_action_spaces):
                space_location = xr.locate_space(
                    space=space, 
                    base_space=self.context.space, 
                    time=time.perf_counter_ns()
                )

                if space_location.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT:
                    role_name = self.role_strings[index]
                    
                    raw_pos = np.array([
                        space_location.pose.position.x,
                        space_location.pose.position.y,
                        space_location.pose.position.z
                    ])
                    
                    raw_quat = np.array([
                        space_location.pose.orientation.x,
                        space_location.pose.orientation.y,
                        space_location.pose.orientation.z,
                        space_location.pose.orientation.w
                    ])

                    # Compute Transform
                    trans_pos, trans_quat = self.compute_rel_transform(raw_pos, raw_quat)
                    
                    frame_data[role_name] = {
                        "pos": trans_pos,
                        "quat": trans_quat, # [x, y, z, w]
                        "raw_pos": raw_pos,
                        "raw_quat": raw_quat
                    }
            
            yield frame_data

    def close(self):
        """Clean up OpenXR resources"""
        if self.context:
            self.context.__exit__(None, None, None)