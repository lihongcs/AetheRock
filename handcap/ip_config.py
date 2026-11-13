import os

WIN_HOST = os.environ.get("HANDCAP_WINDOWS_HOST")
RASPI_HOST = os.environ.get("HANDCAP_RASPI_HOST")
VR_HOST = os.environ.get("HANDCAP_QUEST_HOST")
NTP_HOST = os.environ.get("HANDCAP_NTP_HOST")

UDP_WIN_PORT = 8265 # opened on raspi for sending everything
UDP_PORT = 8265 # opened on raspi for sending everything
POSE_PORT = 12346 # opened on VR, for ee pose
GUIDANCE_PORT = 12347 # opened on VR, for guidance commands

DATA_FPS = 30