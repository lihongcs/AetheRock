from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import cv2
import numpy as np

class BaseCamera(ABC):
    def __init__(self, config_dict: Dict):
        self.config_dict = config_dict
        self._initialize(**config_dict)

    @abstractmethod
    def _initialize(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def read(self) -> Tuple[bool, np.ndarray]:
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def release(self):
        raise NotImplementedError("Subclasses must implement this method")


import sys
import os
import time
from ctypes import *

import cv2
import numpy as np



try:
    sdk_path = os.environ.get("HANDCAP_MVS_SDK_PATH", "/opt/MVS/Samples/aarch64/Python/MvImport")
    if sdk_path not in sys.path:
        sys.path.append(sdk_path)
    from MvCameraControl_class import *

except ImportError:
    print("Failed to import MvCameraControl_class. Ensure the MVS SDK is installed and the path is correct.")


def get_all_mvs_dev_serial():
    """
    获取所有MVS设备的序列号。
    """

    deviceList = MV_CC_DEVICE_INFO_LIST()
    layerType = (MV_GIGE_DEVICE
                 | MV_USB_DEVICE
                 | MV_GENTL_CAMERALINK_DEVICE
                 | MV_GENTL_CXP_DEVICE
                 | MV_GENTL_XOF_DEVICE)

    ret = MvCamera.MV_CC_EnumDevices(layerType, deviceList)
    if ret != 0:
        print("Enum devices fail! ret[0x%x]" % ret)
        sys.exit()

    serial_numbers = []

    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        strSerialNumber = ""

        if mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber += chr(per)
            serial_numbers.append(strSerialNumber)

    return serial_numbers



class MVSCamera(BaseCamera):

    def initialize_sdk(self):
        """
        初始化SDK。
        """
        MvCamera.MV_CC_Initialize()
        sdk_version = MvCamera.MV_CC_GetSDKVersion()
        print("SDK Version: [0x%x]" % sdk_version)

    def enumerate_devices(self):
        """
        枚举可用设备，并将结果存储到 self.deviceList。
        """
        self.deviceList = MV_CC_DEVICE_INFO_LIST()
        layerType = (MV_GIGE_DEVICE
                     | MV_USB_DEVICE
                     | MV_GENTL_CAMERALINK_DEVICE
                     | MV_GENTL_CXP_DEVICE
                     | MV_GENTL_XOF_DEVICE)

        ret = MvCamera.MV_CC_EnumDevices(layerType, self.deviceList)
        if ret != 0:
            print("Enum devices fail! ret[0x%x]" % ret)
            sys.exit()

        if self.deviceList.nDeviceNum == 0:
            print("No device found!")
            sys.exit()

        print("Find %d devices!" % self.deviceList.nDeviceNum)

        self.DevSerialNumbers = []

        for i in range(0, self.deviceList.nDeviceNum):
            strSerialNumber = ""

            mvcc_dev_info = cast(self.deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            # 根据不同接口类型，打印相关信息
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE or mvcc_dev_info.nTLayerType == MV_GENTL_GIGE_DEVICE:
                print("\ngige device: [%d]" % i)
                strModelName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                    if per == 0:
                        break
                    strModelName += chr(per)
                print("device model name: %s" % strModelName)

                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))

            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                print("\nu3v device: [%d]" % i)
                strModelName = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                    if per == 0:
                        break
                    strModelName += chr(per)
                print("device model name: %s" % strModelName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber += chr(per)
                print("user serial number: %s" % strSerialNumber)

            elif mvcc_dev_info.nTLayerType == MV_GENTL_CAMERALINK_DEVICE:
                print("\nCML device: [%d]" % i)
                strModelName = ""
                for per in mvcc_dev_info.SpecialInfo.stCMLInfo.chModelName:
                    if per == 0:
                        break
                    strModelName += chr(per)
                print("device model name: %s" % strModelName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stCMLInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber += chr(per)
                print("user serial number: %s" % strSerialNumber)

            elif mvcc_dev_info.nTLayerType == MV_GENTL_XOF_DEVICE:
                print("\nXoF device: [%d]" % i)
                strModelName = ""
                for per in mvcc_dev_info.SpecialInfo.stXoFInfo.chModelName:
                    if per == 0:
                        break
                    strModelName += chr(per)
                print("device model name: %s" % strModelName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stXoFInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber += chr(per)
                print("user serial number: %s" % strSerialNumber)

            elif mvcc_dev_info.nTLayerType == MV_GENTL_CXP_DEVICE:
                print("\nCXP device: [%d]" % i)
                strModelName = ""
                for per in mvcc_dev_info.SpecialInfo.stCXPInfo.chModelName:
                    if per == 0:
                        break
                    strModelName += chr(per)
                print("device model name: %s" % strModelName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stCXPInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber += chr(per)
                print("user serial number: %s" % strSerialNumber)

            self.DevSerialNumbers.append(strSerialNumber)

    def connect_device(self, index):
        """
        根据索引连接指定设备，并完成打开设备、设置包大小、关闭触发模式等操作。
        """
        # 创建相机实例
        self.cam = MvCamera()

        # 获取设备信息
        self.stDeviceList = cast(self.deviceList.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents

        # 创建句柄
        ret = self.cam.MV_CC_CreateHandle(self.stDeviceList)
        if ret != 0:
            print("create handle fail! ret[0x%x]" % ret)
            sys.exit()

        # 打开设备
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print("open device fail! ret[0x%x]" % ret)
            sys.exit()

        # 若为GigE接口，则设置最佳包大小
        if self.stDeviceList.nTLayerType == MV_GIGE_DEVICE or self.stDeviceList.nTLayerType == MV_GENTL_GIGE_DEVICE:
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret != 0:
                    print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
            else:
                print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

        # 设置触发模式为off
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print("set trigger mode fail! ret[0x%x]" % ret)
            sys.exit()

        # 获取数据包大小
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            print("get payload size fail! ret[0x%x]" % ret)
            sys.exit()

        self.nPayloadSize = stParam.nCurValue
        self.data_buf = (c_ubyte * self.nPayloadSize)()

    ################## abstract methods ##################

    def _initialize(self, serial, exposure):
        self.initialize_sdk()
        self.enumerate_devices()

        if not serial:
            if len(self.DevSerialNumbers) != 1:
                raise ValueError("Set HANDCAP_WRIST_CAMERA_SERIAL to select one of the detected MVS cameras.")
            serial = self.DevSerialNumbers[0]
        if serial not in self.DevSerialNumbers:
            raise ValueError("The configured wrist camera serial was not found among the detected MVS cameras.")
        self.connect_device(self.DevSerialNumbers.index(serial))

        ret = self.cam.MV_CC_StartGrabbing()
        assert ret == 0, f"Start grabbing fail! ret[0x{ret:x}]"

        self.exposure_s = exposure / 1e6  # microseconds → seconds

        ret = self.cam.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF)
        ret = self.cam.MV_CC_SetIntValue("ExposureTime", exposure) # 20000

        # 1. 关闭自动增益模式 (数字 0 代表 Off，也可以用 MV_GAIN_MODE_OFF 如果前面有导入这个常量)
        ret = self.cam.MV_CC_SetEnumValue("GainAuto", 0) 
        
        # 2. 设置具体的增益值，例如设置为 10.0 (您可以根据环境改动这个浮点数，一般范围是 0 到 15~20 多)
        gain_value = 14.0
        ret = self.cam.MV_CC_SetFloatValue("Gain", gain_value)

        ret = self.cam.MV_CC_SetEnumValue("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS)

        ret = self.cam.MV_CC_SetBoolValue("BlackLevelEnable", 1)
        ret = self.cam.MV_CC_SetIntValue("BlackLevel", 240)

        ret = self.cam.MV_CC_SetIntValue("Brightness", 30)

        self.stOutFrame = MV_FRAME_OUT()
        memset(byref(self.stOutFrame), 0, sizeof(self.stOutFrame))



    def read(self):
        ret = self.cam.MV_CC_GetImageBuffer(self.stOutFrame, 1000)
        assert ret == 0, f"Get image buffer fail! ret[0x{ret:x}]"

        # Record timestamp immediately after GetImageBuffer returns,
        # then adjust back by exposure/2 to estimate actual capture midpoint.
        # GetImageBuffer returns after: full exposure + readout + USB transfer.
        # The midpoint of exposure (= effective capture time) is exposure/2 earlier.
        capture_ts = time.time() - self.exposure_s / 2

        nRGBSize = self.stOutFrame.stFrameInfo.nWidth * self.stOutFrame.stFrameInfo.nHeight * 3

        stConvertParam = MV_CC_PIXEL_CONVERT_PARAM_EX()
        memset(byref(stConvertParam), 0, sizeof(stConvertParam))
        stConvertParam.nWidth = self.stOutFrame.stFrameInfo.nWidth
        stConvertParam.nHeight = self.stOutFrame.stFrameInfo.nHeight
        stConvertParam.pSrcData = self.stOutFrame.pBufAddr
        stConvertParam.nSrcDataLen = self.stOutFrame.stFrameInfo.nFrameLen
        stConvertParam.enSrcPixelType = self.stOutFrame.stFrameInfo.enPixelType
        stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed

        stConvertParam.pDstBuffer = (c_ubyte * nRGBSize)()
        stConvertParam.nDstBufferSize = nRGBSize

        ret = self.cam.MV_CC_ConvertPixelTypeEx(stConvertParam)
        if ret != 0:
            print ("convert pixel fail! ret[0x%x]" % ret)
            sys.exit()

        frame_data_ptr = cast(stConvertParam.pDstBuffer, POINTER(c_ubyte))
        frame_data = np.ctypeslib.as_array(frame_data_ptr, shape=(nRGBSize,))
        last_img = frame_data.reshape(self.stOutFrame.stFrameInfo.nHeight, self.stOutFrame.stFrameInfo.nWidth, 3)
        # Must copy: the numpy array is a view over stConvertParam.pDstBuffer
        # which gets freed when this function returns
        last_img = last_img.copy()

        self.cam.MV_CC_FreeImageBuffer(self.stOutFrame)

        return True, last_img, capture_ts
    
    def release(self):
        # return self.cam.release()
        pass
    

if __name__ == "__main__":

    cam = MVSCamera({"serial": os.environ.get("HANDCAP_WRIST_CAMERA_SERIAL"), "exposure": 20000})

    fps = 20
    output_path = "output.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    frame_width = 0
    frame_height = 0

    # Probe one frame to get resolution
    ret, first_img, _ = cam.read()
    assert ret, "Failed to read first frame"
    frame_height, frame_width = first_img.shape[:2]

    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    writer.write(first_img)

    # Pre-fetch the latest frame for smooth display
    ret, current_frame, _ = cam.read()

    for i in range(500):
        ret, frame, _ = cam.read()
        writer.write(frame)

        display = frame.copy()
        cv2.putText(display, "REC", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved: {output_path}")