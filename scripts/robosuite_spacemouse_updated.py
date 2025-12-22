"""Driver class for SpaceMouse controller.

This class provides driver support to SpaceMouse using ctypes to interface
directly with the system's libspnav C library.

Requirements:
    1. Linux system
    2. spacenavd daemon running: `sudo apt install spacenavd && sudo systemctl start spacenavd`
    3. libspnav shared library: `sudo apt install libspnav-dev`
"""

import threading
import time
import ctypes
from ctypes import Structure, Union, c_int, c_uint, c_void_p, CDLL
from collections import namedtuple

import numpy as np

import robosuite.macros as macros
from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix

# ==============================================================================
# Ctypes definitions for libspnav (Avoiding the broken 'spnav' pip package)
# ==============================================================================

SPNAV_EVENT_ANY = 0
SPNAV_EVENT_MOTION = 1
SPNAV_EVENT_BUTTON = 2
xyzTHRESHOLD = 100
rpyTHRESHOLD = 300

class SpnavEventMotion(Structure):
    _fields_ = [
        ("type", c_int),
        ("x", c_int),
        ("y", c_int),
        ("z", c_int),
        ("rx", c_int),
        ("ry", c_int),
        ("rz", c_int),
        ("period", c_uint),
        ("data", c_void_p),
    ]

class SpnavEventButton(Structure):
    _fields_ = [
        ("type", c_int),
        ("press", c_int),
        ("bnum", c_int),
    ]

class SpnavEvent(Union):
    _fields_ = [
        ("type", c_int),
        ("motion", SpnavEventMotion),
        ("button", SpnavEventButton),
    ]

# Try to load the system library
try:
    # Usually located at /usr/lib/x86_64-linux-gnu/libspnav.so
    libspnav = CDLL("libspnav.so.0") 
except OSError:
    try:
        libspnav = CDLL("libspnav.so")
    except OSError:
        raise ImportError(
            "Could not load libspnav.so. "
            "Please run: sudo apt-get install libspnav-dev"
        )

# Define function signatures
libspnav.spnav_open.restype = c_int
libspnav.spnav_close.restype = c_int
libspnav.spnav_poll_event.argtypes = [ctypes.POINTER(SpnavEvent)]
libspnav.spnav_poll_event.restype = c_int

# ==============================================================================
# End Ctypes definitions
# ==============================================================================

AxisSpec = namedtuple("AxisSpec", ["channel", "byte1", "byte2", "scale"])

def scale_to_control(x, axis_scale=500.0, min_v=-1.0, max_v=1.0):
    """
    Normalize raw readings to target range.
    """
    x = x / axis_scale
    x = min(max(x, min_v), max_v)
    return x

class SpaceMouse(Device):
    """
    A driver class for SpaceMouse using direct libspnav bindings.
    """

    def __init__(
        self,
        vendor_id=macros.SPACEMOUSE_VENDOR_ID,  # Unused in spnav mode
        product_id=macros.SPACEMOUSE_PRODUCT_ID, # Unused in spnav mode
        pos_sensitivity=1.0,
        rot_sensitivity=1.0,
    ):
        print("Opening SpaceMouse via system libspnav...")
        
        # 0 means connection to local spacenavd
        ret = libspnav.spnav_open()
        if ret == -1:
            print("Failed to connect to spacenavd daemon.")
            print("Please ensure it is running: sudo systemctl start spacenavd")
            raise IOError("spnav_open failed")

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        # 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self._display_controls()

        self.single_click_and_hold = False

        self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._reset_state = 0
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self._enabled = False

        # launch a new listener thread
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    @staticmethod
    def _display_controls():
        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Control", "Command")
        print_command("Right button", "reset simulation")
        print_command("Left button (hold)", "close gripper")
        print_command("Move mouse laterally", "move arm horizontally in x-y plane")
        print_command("Move mouse vertically", "move arm vertically")
        print_command("Twist mouse about an axis", "rotate arm about a corresponding axis")
        print_command("ESC", "quit")
        print("")

    def _reset_internal_state(self):
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0
        self._control = np.zeros(6)
        self.single_click_and_hold = False

    def start_control(self):
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        dpos = self.control[:3] * 0.005 * self.pos_sensitivity
        roll, pitch, yaw = self.control[3:] * 0.005 * self.rot_sensitivity

        drot1 = rotation_matrix(angle=-pitch, direction=[1.0, 0, 0], point=None)[:3, :3]
        drot2 = rotation_matrix(angle=roll, direction=[0, 1.0, 0], point=None)[:3, :3]
        drot3 = rotation_matrix(angle=yaw, direction=[0, 0, 1.0], point=None)[:3, :3]

        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))

        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=np.array([roll, pitch, yaw]),
            grasp=self.control_gripper,
            reset=self._reset_state,
        )

    def run(self):
        """Listener method using ctypes to poll libspnav."""
        
        event = SpnavEvent()

        while True:
            if self._enabled:
                # spnav_poll_event returns non-zero if an event was retrieved
                while libspnav.spnav_poll_event(ctypes.byref(event)) != 0:
                    
                    if event.type == SPNAV_EVENT_MOTION:
                        # Access fields via the union structure
                        motion = event.motion
                        
                        if abs(motion.x) < xyzTHRESHOLD:
                            self.x = 0
                        if abs(motion.y) < xyzTHRESHOLD:
                            self.y = 0
                        if abs(motion.z) < xyzTHRESHOLD:
                            self.z = 0
                        if abs(motion.rx) < rpyTHRESHOLD:
                            self.rx = 0
                        else:
                            self.rx *= (500 - rpyTHRESHOLD)/500
                        if abs(motion.ry) < rpyTHRESHOLD:
                            self.ry = 0
                        else:
                            self.ry *= (500 - rpyTHRESHOLD)/500
                        if abs(motion.rz) < rpyTHRESHOLD:
                            self.rz = 0
                        else:
                            self.rz *= (500 - rpyTHRESHOLD)/500
                        # Mapping Spacemouse axes to Robosuite axes
                        # Adjust signs if movement is inverted
                        # print(f"x: {motion.x}\ny: {motion.y}\nz: {motion.z}")
                        # print(f"rx: {motion.rx}\nry: {motion.ry}\nrz: {motion.rz}")
                        
                        self.x = scale_to_control(motion.z) * -1.0
                        self.y = scale_to_control(motion.x)
                        self.z = scale_to_control(motion.y)
                        self.roll = scale_to_control(motion.rx)
                        self.pitch = scale_to_control(motion.rz) * -1.0
                        self.yaw = scale_to_control(motion.ry) * -1.0
                        

                        self._control = [
                            self.x,
                            self.y,
                            self.z,
                            self.roll,
                            self.pitch,
                            self.yaw,
                        ]

                    elif event.type == SPNAV_EVENT_BUTTON:
                        btn = event.button
                        is_pressed = (btn.press == 1)
                        
                        # Button 0 -> Left (Grasp)
                        if btn.bnum == 0 and is_pressed:
                            self.single_click_and_hold = 1 - self.single_click_and_hold
                        
                        # Button 1 -> Right (Reset)
                        if btn.bnum == 1 and is_pressed:
                            self._reset_state = 1
                            self._enabled = False
                            self._reset_internal_state()

            # Small sleep to prevent CPU hogging
            time.sleep(0.005)

    @property
    def control(self):
        return np.array(self._control)

    @property
    def control_gripper(self):
        if self.single_click_and_hold:
            return 1.0
        return 0

    def __del__(self):
        try:
            libspnav.spnav_close()
        except:
            pass

if __name__ == "__main__":
    try:
        space_mouse = SpaceMouse()
        space_mouse.start_control()
        print("Listening for input... Press Ctrl+C to stop.")
        while True:
            # Uncomment below to debug raw values
            # print(f"Control: {space_mouse.control} | Gripper: {space_mouse.control_gripper}")
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nStopped.")