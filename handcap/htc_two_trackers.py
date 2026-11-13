import os
import xr
import time
import msvcrt
import ctypes
import traceback
import threading
import numpy as np
from argparse import ArgumentParser
import scipy.spatial.transform as st
from ctypes import cast, byref, POINTER
from scipy.spatial.transform import Rotation

try:
    from Postprocess.umi_utils.pose_util import pose_to_mat, mat_to_certain_pose_type
except ImportError:
    pass 

collecting = False
running = True

def is_windows_elevated():
    if os.name != "nt":
        return False
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False

def get_windows_openxr_runtime():
    if os.name != "nt":
        return None
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Khronos\OpenXR",
        ) as key:
            return winreg.QueryValueEx(key, "ActiveRuntime")[0]
    except Exception:
        return None

def print_openxr_preflight():
    if os.name != "nt":
        return
    api_layer_path = os.environ.get("XR_API_LAYER_PATH")
    active_runtime = get_windows_openxr_runtime()
    elevated = is_windows_elevated()
    print("[OpenXR Preflight]")
    print(f"  elevated/admin: {elevated}")
    print(f"  ActiveRuntime: {active_runtime or '<not found>'}")
    print(f"  XR_API_LAYER_PATH: {api_layer_path or '<not set>'}")
    if elevated and api_layer_path:
        print("  WARNING: OpenXR loader ignores XR_API_LAYER_PATH in elevated/admin shells. Run this script from a normal Anaconda Prompt/PowerShell.")
    if not active_runtime:
        print("  WARNING: No active OpenXR runtime is registered. Set SteamVR/VIVE/OpenXR runtime as the active OpenXR runtime first.")

def quat_to_rot(quat_pos):
    shape = quat_pos.shape[:-1]
    pose = np.zeros(shape+(6,), dtype=quat_pos.dtype)
    rot = st.Rotation.from_quat(quat_pos[...,3:])
    pos = quat_pos[...,:3]
    pose[...,:3] = pos
    pose[...,3:] = rot.as_rotvec()
    return pose

def compute_pose_transform(raw_pose):
    try:
        pose_mat = pose_to_mat(quat_to_rot(raw_pose))
        additional_transform_matrix = np.eye(4)
        additional_R_y = Rotation.from_euler('y', 90, degrees=True).as_matrix()
        additional_R_z = Rotation.from_euler('z', 180, degrees=True).as_matrix()
        transform_mat =  additional_R_y @ additional_R_z
        additional_transform_matrix[:3, :3] = transform_mat
        pose_mat_out = pose_mat @ additional_transform_matrix
        return mat_to_certain_pose_type(pose_mat_out, pose_type="rotvec")
    except NameError:
        return None

def keyboard_listener():
    global collecting, running
    while running:
        try:
            has_key = msvcrt.kbhit()
        except OSError:
            print("Keyboard control is unavailable in this process. Remote control remains active.")
            break

        if has_key:
            key = msvcrt.getch()
            if key == b'\x1b':  # ESC
                collecting = False
                running = False
                print("ESC pressed. Exiting.")
            elif key == b' ':  # Space
                collecting = not collecting
                state = "Started" if collecting else "Stopped"
                print(f"Space pressed. {state} collection.")
        time.sleep(0.01)

def control_file_listener(stop_file):
    global collecting, running
    while running:
        if stop_file and os.path.exists(stop_file):
            collecting = False
            running = False
            try:
                os.remove(stop_file)
            except OSError:
                pass
            print(f"Remote stop requested by {stop_file}. Exiting.")
            break
        time.sleep(0.05)

class ContextObject(object):
    def __init__(
            self,
            instance_create_info: xr.InstanceCreateInfo = xr.InstanceCreateInfo(),
            session_create_info: xr.SessionCreateInfo = xr.SessionCreateInfo(),
            reference_space_create_info: xr.ReferenceSpaceCreateInfo = xr.ReferenceSpaceCreateInfo(),
            view_configuration_type: xr.ViewConfigurationType = xr.ViewConfigurationType.PRIMARY_STEREO,
            environment_blend_mode=xr.EnvironmentBlendMode.OPAQUE,
            form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY,
    ):
        self._instance_create_info = instance_create_info
        self.instance = None
        self._session_create_info = session_create_info
        self.session = None
        self.session_state = xr.SessionState.IDLE
        self._reference_space_create_info = reference_space_create_info
        self.view_configuration_type = view_configuration_type
        self.environment_blend_mode = environment_blend_mode
        self.form_factor = form_factor
        self.graphics = None
        self.graphics_binding_pointer = None
        self.action_sets = []
        self.render_layers = []
        self.swapchains = []
        self.swapchain_image_ptr_buffers = []
        self.swapchain_image_buffers = []  # Keep alive
        self.exit_render_loop = False
        self.request_restart = False  # TODO: do like hello_xr
        self.session_is_running = False

    def __enter__(self):
        self.instance = xr.create_instance(
            create_info=self._instance_create_info,
        )
        self.system_id = xr.get_system(
            instance=self.instance,
            get_info=xr.SystemGetInfo(
                form_factor=self.form_factor,
            ),
        )

        if self._session_create_info.next is not None:
            self.graphics_binding_pointer = self._session_create_info.next

        self._session_create_info.system_id = self.system_id
        self.session = xr.create_session(
            instance=self.instance,
            create_info=self._session_create_info,
        )
        self.space = xr.create_reference_space(
            session=self.session,
            create_info=self._reference_space_create_info
        )
        self.default_action_set = xr.create_action_set(
            instance=self.instance,
            create_info=xr.ActionSetCreateInfo(
                action_set_name="default_action_set",
                localized_action_set_name="Default Action Set",
                priority=0,
            ),
        )
        self.action_sets.append(self.default_action_set)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.default_action_set is not None:
            xr.destroy_action_set(self.default_action_set)
            self.default_action_set = None
        if self.space is not None:
            xr.destroy_space(self.space)
            self.space = None
        if self.session is not None:
            xr.destroy_session(self.session)
            self.session = None
        if self.graphics is not None:
            self.graphics.destroy()
            self.graphics = None
        if self.instance is not None:
            xr.destroy_instance(self.instance)
            self.instance = None

    def frame_loop(self):
        xr.attach_session_action_sets(
            session=self.session,
            attach_info=xr.SessionActionSetsAttachInfo(
                count_action_sets=len(self.action_sets),
                action_sets=(xr.ActionSet * len(self.action_sets))(
                    *self.action_sets
                )
            ),
        )
        while True:
            if not running:
                break
            self.exit_render_loop = False
            self.poll_xr_events()
            if self.exit_render_loop:
                break
            if self.session_is_running:
                if self.session_state in (
                        xr.SessionState.READY,
                        xr.SessionState.SYNCHRONIZED,
                        xr.SessionState.VISIBLE,
                        xr.SessionState.FOCUSED,
                ):
                    frame_state = xr.wait_frame(self.session)
                    xr.begin_frame(self.session)
                    self.render_layers = []

                    yield frame_state

                    xr.end_frame(
                        self.session,
                        frame_end_info=xr.FrameEndInfo(
                            display_time=frame_state.predicted_display_time,
                            environment_blend_mode=self.environment_blend_mode,
                            layers=self.render_layers,
                        )
                    )
            else:
                # Throttle loop since xrWaitFrame won't be called.
                time.sleep(0.250)

    def poll_xr_events(self):
        self.exit_render_loop = False
        self.request_restart = False
        while True:
            try:
                event_buffer = xr.poll_event(self.instance)
                try:
                    event_type = xr.StructureType(event_buffer.type)
                except ValueError as e:
                    print(f"Unknown event type: {event_buffer.type}, skipping...")
                    continue
                # event_type = xr.StructureType(event_buffer.type)
                if event_type == xr.StructureType.EVENT_DATA_INSTANCE_LOSS_PENDING:
                    # still handle rest of the events instead of immediately quitting
                    self.exit_render_loop = True
                    self.request_restart = True
                elif event_type == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED \
                        and self.session is not None:
                    event = cast(
                        byref(event_buffer),
                        POINTER(xr.EventDataSessionStateChanged)).contents
                    self.session_state = xr.SessionState(event.state)
                    if self.session_state == xr.SessionState.READY:
                        xr.begin_session(
                            session=self.session,
                            begin_info=xr.SessionBeginInfo(
                                self.view_configuration_type,
                            ),
                        )
                        self.session_is_running = True
                    elif self.session_state == xr.SessionState.STOPPING:
                        self.session_is_running = False
                        xr.end_session(self.session)
                    elif self.session_state == xr.SessionState.EXITING:
                        self.exit_render_loop = True
                        self.request_restart = False
                    elif self.session_state == xr.SessionState.LOSS_PENDING:
                        self.exit_render_loop = True
                        self.request_restart = True
                elif event_type == xr.StructureType.EVENT_DATA_VIVE_TRACKER_CONNECTED_HTCX:
                    vive_tracker_connected = cast(byref(event_buffer), POINTER(xr.EventDataViveTrackerConnectedHTCX)).contents
                    paths = vive_tracker_connected.paths.contents
                    persistent_path_str = xr.path_to_string(self.instance, paths.persistent_path)
                    if paths.role_path != xr.NULL_PATH:
                        role_path_str = xr.path_to_string(self.instance, paths.role_path)
                    else:
                        pass
                elif event_type == xr.StructureType.EVENT_DATA_INTERACTION_PROFILE_CHANGED:
                    pass
            except xr.EventUnavailable:
                break

    def view_loop(self, frame_state):
        if frame_state.should_render:
            layer = xr.CompositionLayerProjection(space=self.space)
            view_state, views = xr.locate_views(
                session=self.session,
                view_locate_info=xr.ViewLocateInfo(
                    view_configuration_type=self.view_configuration_type,
                    display_time=frame_state.predicted_display_time,
                    space=self.space,
                )
            )
            num_views = len(views)
            projection_layer_views = tuple(xr.CompositionLayerProjectionView() for _ in range(num_views))

            vsf = view_state.view_state_flags
            if (vsf & xr.VIEW_STATE_POSITION_VALID_BIT == 0
                    or vsf & xr.VIEW_STATE_ORIENTATION_VALID_BIT == 0):
                return  # There are no valid tracking poses for the views.
            for view_index, view in enumerate(views):
                view_swapchain = self.swapchains[view_index]
                swapchain_image_index = xr.acquire_swapchain_image(
                    swapchain=view_swapchain.handle,
                    acquire_info=xr.SwapchainImageAcquireInfo(),
                )
                xr.wait_swapchain_image(
                    swapchain=view_swapchain.handle,
                    wait_info=xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION),
                )
                layer_view = projection_layer_views[view_index]
                assert layer_view.type == xr.StructureType.COMPOSITION_LAYER_PROJECTION_VIEW
                layer_view.pose = view.pose
                layer_view.fov = view.fov
                layer_view.sub_image.swapchain = view_swapchain.handle
                layer_view.sub_image.image_rect.offset[:] = [0, 0]
                layer_view.sub_image.image_rect.extent[:] = [
                    view_swapchain.width, view_swapchain.height, ]
                swapchain_image_ptr = self.swapchain_image_ptr_buffers[view_index][swapchain_image_index]
                swapchain_image = cast(swapchain_image_ptr, POINTER(xr.SwapchainImageOpenGLKHR)).contents
                assert layer_view.sub_image.image_array_index == 0  # texture arrays not supported.
                color_texture = swapchain_image.image
                self.graphics.begin_frame(layer_view, color_texture)

                yield view

                self.graphics.end_frame()
                xr.release_swapchain_image(
                    swapchain=view_swapchain.handle,
                    release_info=xr.SwapchainImageReleaseInfo()
                )
            layer.views = projection_layer_views
            self.render_layers.append(byref(layer))


def main(args):
    global collecting, running
    collecting = bool(getattr(args, "collect_on_start", False))
    running = True
    target_roles = [
        role.strip()
        for role in getattr(args, "tracker_roles", "right_elbow,left_elbow").split(",")
        if role.strip()
    ]
    
    save_dir = f"data/htc_only_{args.cap_id}"
    os.makedirs(save_dir, exist_ok=True)
    
    tracker_files = {}
    tracker_file_paths = {}
    tracker_counts = {role: 0 for role in target_roles}
    for role in target_roles:
        tracker_file_path = os.path.join(save_dir, f"{role}.csv")
        tracker_file_paths[role] = tracker_file_path
        tracker_file = open(tracker_file_path, "w", buffering=1)
        tracker_file.write("timestamp,pos_x,pos_y,pos_z,ori_x,ori_y,ori_z,ori_w\n")
        tracker_files[role] = tracker_file

    print("******** Control Instructions *********************")
    print("******** Press 'SPACE' to start/stop collection ***")
    print("******** Press 'ESC' to quit ************************")
    print(f"******** Saving tracker roles: {', '.join(target_roles)}")
    if collecting:
        print("******** Remote mode: collection starts immediately *")
    
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()

    stop_file = getattr(args, "stop_file", "")
    if stop_file:
        control_thread = threading.Thread(target=control_file_listener, args=(stop_file,), daemon=True)
        control_thread.start()

    print_openxr_preflight()
    print("Creating OpenXR context...")
    
    try:
        with ContextObject(
            instance_create_info=xr.InstanceCreateInfo(
                enabled_extension_names=[
                    xr.MND_HEADLESS_EXTENSION_NAME,
                    "XR_HTCX_vive_tracker_interaction",
                ],
            ),
        ) as context:
            instance = context.instance
            session = context.session
            print("OpenXR context created successfully!")

            enumerateViveTrackerPathsHTCX = cast(
                xr.get_instance_proc_addr(instance, "xrEnumerateViveTrackerPathsHTCX"),
                xr.PFN_xrEnumerateViveTrackerPathsHTCX
            )

            role_strings = [
                "handheld_object", "left_foot", "right_foot", "left_shoulder", "right_shoulder",
                "left_elbow", "right_elbow", "left_knee", "right_knee", "waist", "chest",
                "camera", "keyboard",
            ]
            role_path_strings = [f"/user/vive_tracker_htcx/role/{role}" for role in role_strings]
            role_paths = (xr.Path * len(role_path_strings))(*[xr.string_to_path(instance, role_string) for role_string in role_path_strings])
            pose_action = xr.create_action(
                action_set=context.default_action_set,
                create_info=xr.ActionCreateInfo(
                    action_type=xr.ActionType.POSE_INPUT,
                    action_name="tracker_pose",
                    localized_action_name="Tracker Pose",
                    count_subaction_paths=len(role_paths),
                    subaction_paths=role_paths,
                ),
            )
            suggested_binding_paths = (xr.ActionSuggestedBinding * len(role_path_strings))(
                *[xr.ActionSuggestedBinding(pose_action, xr.string_to_path(instance, f"{role_path_string}/input/grip/pose")) for role_path_string in role_path_strings]
            )
            xr.suggest_interaction_profile_bindings(instance=instance, suggested_bindings=xr.InteractionProfileSuggestedBinding(
                interaction_profile=xr.string_to_path(instance, "/interaction_profiles/htc/vive_tracker_htcx"),
                count_suggested_bindings=len(suggested_binding_paths), suggested_bindings=suggested_binding_paths,
            ))
            tracker_action_spaces = (xr.Space * len(role_paths))(
                *[xr.create_action_space(session=session, create_info=xr.ActionSpaceCreateInfo(action=pose_action, subaction_path=role_path)) for role_path in role_paths]
            )

            n_paths = ctypes.c_uint32(0)
            result = enumerateViveTrackerPathsHTCX(instance, 0, byref(n_paths), None)
            if xr.check_result(result).is_exception():
                raise result
            
            _n_paths = ctypes.c_uint32(0)
            result = enumerateViveTrackerPathsHTCX(instance, 0, byref(_n_paths), None)
            if xr.check_result(result).is_exception():
                raise result
            _vive_tracker_paths = (xr.ViveTrackerPathsHTCX * _n_paths.value)(*([xr.ViveTrackerPathsHTCX()] * _n_paths.value))
            result = enumerateViveTrackerPathsHTCX(instance, _n_paths, byref(_n_paths), _vive_tracker_paths)
            if xr.check_result(result).is_exception():
                raise result
            _tracker_action_spaces = (xr.Space * len(_vive_tracker_paths))(
                *[xr.create_action_space(session=session, create_info=xr.ActionSpaceCreateInfo(
                    action=pose_action, subaction_path=tp.paths.principal_path))
                    for tp in _vive_tracker_paths]
            )
            
            print("Starting main loop - entering frame_loop()...")
            
            last_print = time.time()
            step_idx = 0
            
            for frame_state in context.frame_loop():
                if not running:
                    break

                if context.session_state != xr.SessionState.FOCUSED:
                    time.sleep(0.5)
                    continue

                active_action_set = xr.ActiveActionSet(action_set=context.default_action_set, subaction_path=xr.NULL_PATH)
                xr.sync_actions(session=session, sync_info=xr.ActionsSyncInfo(count_active_action_sets=1, active_action_sets=ctypes.pointer(active_action_set)))
                
                trackers_data = []
                
                for index, space in enumerate(tracker_action_spaces):
                    space_location = xr.locate_space(space=space, base_space=context.space, time=time.perf_counter_ns())
                    if space_location.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT:
                        role = role_strings[index]
                        position = [
                            space_location.pose.position.x,
                            space_location.pose.position.y,
                            space_location.pose.position.z
                        ]
                        orientation = [
                            space_location.pose.orientation.x,
                            space_location.pose.orientation.y,
                            space_location.pose.orientation.z,
                            space_location.pose.orientation.w,
                        ]
                        trackers_data.append((role, position, orientation))
                
                if collecting and len(trackers_data) > 0:
                    ts = time.time()
                    for role, pos, ori in trackers_data:
                        tracker_file = tracker_files.get(role)
                        if tracker_file is None:
                            continue
                        tracker_file.write(f"{ts},{pos[0]},{pos[1]},{pos[2]},{ori[0]},{ori[1]},{ori[2]},{ori[3]}\n")
                        tracker_counts[role] += 1
                    step_idx += 1
                
                if time.time() - last_print > 1.0:
                    last_print = time.time()
                    status = "collecting" if collecting else "paused"
                    if len(trackers_data) > 0:
                        visible_roles = [role for role, _, _ in trackers_data if role in tracker_files]
                        print(f"[{status}] Tracking {len(trackers_data)} trackers. target visible: {visible_roles or '<none>'}")
                    else:
                        print(f"[{status}] No tracker data")

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        for tracker_file in tracker_files.values():
            tracker_file.close()
        print(f"Saved {step_idx} frame steps to {save_dir}")
        for role in target_roles:
            print(f"  {role}: {tracker_counts.get(role, 0)} rows -> {tracker_file_paths.get(role)}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cap_id", type=int, default=112501)
    parser.add_argument("--collect-on-start", action="store_true")
    parser.add_argument("--stop-file", type=str, default="")
    parser.add_argument("--tracker-roles", type=str, default="right_elbow,left_elbow")
    args = parser.parse_args()
    main(args)
