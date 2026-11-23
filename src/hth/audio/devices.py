"""Windows Core Audio device enumeration and discovery service.

Phase 4 implementation:
- Real WASAPI / MMDevice API enumeration using ctypes.
- Discovers stable, persistent Endpoint GUID identifiers.
- Detects physical mic, output, headphones, and VB-CABLE endpoints by exact name.
- Reads Windows Console and Communications default roles via IMMDeviceEnumerator.
- Never mutates Windows defaults automatically.
- Includes a pluggable AudioDeviceProvider interface for testing with fake device inventories.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import HRESULT, POINTER, Structure, c_ulong, c_wchar_p, wintypes
from typing import Protocol

from hth.core.logging import get_logger
from hth.domain.api_contracts import DeviceInfo, DeviceType
from hth.domain.audio_contracts import (
    AudioDeviceInventory,
    AudioEndpointInfo,
    EndpointFormFactor,
)

logger = get_logger("hth.audio.devices")

# ---------------------------------------------------------------------------
# COM & MMDevice Constants & GUIDs
# ---------------------------------------------------------------------------
CLSCTX_ALL = 23
DEVICE_STATE_ACTIVE = 0x00000001

# DataFlow enum
E_RENDER = 0
E_CAPTURE = 1

# ERole enum
E_CONSOLE = 0
E_MULTIMEDIA = 1
E_COMMUNICATIONS = 2

# Endpoint FormFactor enum mapping
FORM_FACTOR_MAP: dict[int, EndpointFormFactor] = {
    0: EndpointFormFactor.UNKNOWN,  # RemoteNetworkDevice
    1: EndpointFormFactor.SPEAKERS,  # Speakers
    2: EndpointFormFactor.LINE_LEVEL,  # LineLevel
    3: EndpointFormFactor.HEADPHONES,  # Headphones
    4: EndpointFormFactor.MICROPHONE,  # Microphone
    5: EndpointFormFactor.HEADSET,  # Headset
    6: EndpointFormFactor.HANDSET,  # Handset
    7: EndpointFormFactor.DIGITAL_AUDIO,  # DigitalAudioDisplayDevice
}

# Known VB-CABLE Friendly Name signatures
VB_CABLE_INPUT_NAME = "CABLE Input (VB-Audio Virtual Cable)"
VB_CABLE_OUTPUT_NAME = "CABLE Output (VB-Audio Virtual Cable)"


class GUID(Structure):
    _fields_ = [
        ("Data1", c_ulong),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", wintypes.BYTE * 8),
    ]

    def __init__(self, guid_str: str = "") -> None:
        super().__init__()
        if not guid_str:
            return
        # Format: {xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx}
        clean = guid_str.strip("{}")
        parts = clean.split("-")
        self.Data1 = int(parts[0], 16)
        self.Data2 = int(parts[1], 16)
        self.Data3 = int(parts[2], 16)
        b = bytes.fromhex(parts[3] + parts[4])
        for i in range(8):
            self.Data4[i] = b[i]


# Standard MMDevice GUIDs
CLSID_MMDeviceEnumerator = GUID("{BCDE0395-E52F-467C-8E3D-C4579291692E}")
IID_IMMDeviceEnumerator = GUID("{A95664D2-9614-4F35-A746-DE8DB63617E6}")


class PROPERTYKEY(Structure):
    _fields_ = [
        ("fmtid", GUID),
        ("pid", wintypes.DWORD),
    ]


class PROPVARIANT_UNION(ctypes.Union):
    _fields_ = [
        ("pwszVal", c_wchar_p),
        ("uiVal", wintypes.UINT),
        ("ulVal", wintypes.ULONG),
    ]


VARTYPE = ctypes.c_ushort


class PROPVARIANT(Structure):
    _fields_ = [
        ("vt", VARTYPE),
        ("wReserved1", wintypes.WORD),
        ("wReserved2", wintypes.WORD),
        ("wReserved3", wintypes.WORD),
        ("u", PROPVARIANT_UNION),
    ]


VT_LPWSTR = 31
VT_UI4 = 19

# PKEY_Device_FriendlyName: {a45c254e-df1c-4efd-8020-67d146a850e0}, 2
PKEY_Device_FriendlyName = PROPERTYKEY(GUID("{a45c254e-df1c-4efd-8020-67d146a850e0}"), 2)
# PKEY_DeviceInterface_FriendlyName: {b3f8fa53-0004-438e-9003-51a46e139bfc}, 6
PKEY_DeviceInterface_FriendlyName = PROPERTYKEY(GUID("{b3f8fa53-0004-438e-9003-51a46e139bfc}"), 6)
# PKEY_AudioEndpoint_FormFactor: {1da5d803-d492-4edd-8c23-e0c0ffee7f0e}, 0
PKEY_AudioEndpoint_FormFactor = PROPERTYKEY(GUID("{1da5d803-d492-4edd-8c23-e0c0ffee7f0e}"), 0)


# ---------------------------------------------------------------------------
# Provider Interface
# ---------------------------------------------------------------------------
class AudioDeviceProvider(Protocol):
    """Abstract interface for audio device discovery to enable fake test inventories."""

    def get_inventory(self) -> AudioDeviceInventory:
        """Query and return the complete audio device inventory."""
        ...


# ---------------------------------------------------------------------------
# Windows MMDevice Implementation (ctypes)
# ---------------------------------------------------------------------------
def _release_com_ptr(ptr: ctypes.c_void_p) -> None:
    if ptr:
        try:
            vtable = ctypes.cast(ptr, POINTER(POINTER(ctypes.c_void_p))).contents
            release_func = ctypes.WINFUNCTYPE(c_ulong, ctypes.c_void_p)(vtable[2])
            release_func(ptr)
        except Exception:
            pass


class WindowsMMDeviceProvider:
    """Enumerate Windows Core Audio endpoints natively using COM."""

    def __init__(self) -> None:
        self._ole32 = None
        if sys.platform == "win32":
            try:
                self._ole32 = ctypes.oledll.ole32
                self._prop_free = ctypes.windll.ole32.PropVariantClear
            except Exception as e:
                logger.warning("Failed to load ole32.dll for MMDevice enumeration", error=str(e))

    @staticmethod
    def _release_com(ptr: ctypes.c_void_p) -> None:
        if ptr:
            try:
                vtable = ctypes.cast(ptr, POINTER(POINTER(ctypes.c_void_p))).contents
                release_func = ctypes.WINFUNCTYPE(c_ulong, ctypes.c_void_p)(vtable[2])
                release_func(ptr)
            except Exception:
                pass

    def _query_endpoint_details(
        self, device_ptr: ctypes.c_void_p, data_flow: str
    ) -> AudioEndpointInfo | None:
        if not device_ptr:
            return None

        # Vtable indices for IMMDevice:
        # 0: QueryInterface, 1: AddRef, 2: Release
        # 3: Activate, 4: OpenPropertyStore, 5: GetId, 6: GetState
        vtable = ctypes.cast(device_ptr, POINTER(POINTER(ctypes.c_void_p))).contents

        # GetId (index 5)
        # HRESULT GetId([out] LPWSTR *ppstrId);
        get_id_func = ctypes.WINFUNCTYPE(HRESULT, ctypes.c_void_p, POINTER(c_wchar_p))(vtable[5])
        p_id = c_wchar_p()
        hr = get_id_func(device_ptr, ctypes.byref(p_id))
        if hr != 0 or not p_id.value:
            return None
        endpoint_id = p_id.value

        # OpenPropertyStore (index 4)
        # HRESULT OpenPropertyStore([in] DWORD stgmAccess, [out] IPropertyStore **ppProperties);
        # STGM_READ = 0
        open_store_func = ctypes.WINFUNCTYPE(
            HRESULT, ctypes.c_void_p, wintypes.DWORD, POINTER(ctypes.c_void_p)
        )(vtable[4])
        store_ptr = ctypes.c_void_p()
        hr = open_store_func(device_ptr, 0, ctypes.byref(store_ptr))
        if hr != 0 or not store_ptr:
            return AudioEndpointInfo(
                id=endpoint_id,
                friendly_name="Unknown Endpoint",
                data_flow=data_flow,  # type: ignore[arg-type]
            )

        store_vtable = ctypes.cast(store_ptr, POINTER(POINTER(ctypes.c_void_p))).contents
        # IPropertyStore:
        # 0: QI, 1: AddRef, 2: Release, 3: GetCount, 4: GetAt, 5: GetValue, 6: SetValue, 7: Commit
        get_value_func = ctypes.WINFUNCTYPE(
            HRESULT, ctypes.c_void_p, POINTER(PROPERTYKEY), POINTER(PROPVARIANT)
        )(store_vtable[5])
        release_store_func = ctypes.WINFUNCTYPE(c_ulong, ctypes.c_void_p)(store_vtable[2])

        friendly_name = "Unknown Audio Device"
        adapter_name = ""
        form_factor = EndpointFormFactor.UNKNOWN

        # 1. Read PKEY_Device_FriendlyName
        pv = PROPVARIANT()
        if get_value_func(store_ptr, ctypes.byref(PKEY_Device_FriendlyName), ctypes.byref(pv)) == 0:
            if pv.vt == VT_LPWSTR and pv.u.pwszVal:
                friendly_name = pv.u.pwszVal
            self._prop_free(ctypes.byref(pv))

        # 2. Read PKEY_DeviceInterface_FriendlyName
        pv2 = PROPVARIANT()
        if (
            get_value_func(
                store_ptr, ctypes.byref(PKEY_DeviceInterface_FriendlyName), ctypes.byref(pv2)
            )
            == 0
        ):
            if pv2.vt == VT_LPWSTR and pv2.u.pwszVal:
                adapter_name = pv2.u.pwszVal
            self._prop_free(ctypes.byref(pv2))

        # 3. Read PKEY_AudioEndpoint_FormFactor
        pv3 = PROPVARIANT()
        if (
            get_value_func(
                store_ptr, ctypes.byref(PKEY_AudioEndpoint_FormFactor), ctypes.byref(pv3)
            )
            == 0
        ):
            if pv3.vt == VT_UI4:
                form_factor = FORM_FACTOR_MAP.get(pv3.u.ulVal, EndpointFormFactor.UNKNOWN)
            self._prop_free(ctypes.byref(pv3))

        release_store_func(store_ptr)

        is_vb_in = "cable input" in friendly_name.lower() or "cable input" in adapter_name.lower()
        is_vb_out = (
            "cable output" in friendly_name.lower() or "cable output" in adapter_name.lower()
        )
        is_phys = not (is_vb_in or is_vb_out or "virtual" in friendly_name.lower())

        return AudioEndpointInfo(
            id=endpoint_id,
            friendly_name=friendly_name,
            adapter_name=adapter_name,
            data_flow=data_flow,  # type: ignore[arg-type]
            form_factor=form_factor,
            is_vb_cable_input=is_vb_in,
            is_vb_cable_output=is_vb_out,
            is_physical=is_phys,
        )

    def get_inventory(self) -> AudioDeviceInventory:
        if sys.platform != "win32" or not self._ole32:
            return AudioDeviceInventory()

        # CoInitialize
        hr = self._ole32.CoInitialize(None)
        co_initialized = hr in (0, 1)  # S_OK or S_FALSE

        inventory = AudioDeviceInventory()

        try:
            enum_ptr = ctypes.c_void_p()
            hr = self._ole32.CoCreateInstance(
                ctypes.byref(CLSID_MMDeviceEnumerator),
                None,
                CLSCTX_ALL,
                ctypes.byref(IID_IMMDeviceEnumerator),
                ctypes.byref(enum_ptr),
            )
            if hr != 0 or not enum_ptr:
                logger.error("Failed to create IMMDeviceEnumerator", hr=hr)
                return inventory

            enum_vtable = ctypes.cast(enum_ptr, POINTER(POINTER(ctypes.c_void_p))).contents
            # IMMDeviceEnumerator:
            # 0: QI, 1: AddRef, 2: Release
            # 3: EnumAudioEndpoints
            # 4: GetDefaultAudioEndpoint
            # 5: GetDevice
            # 6: RegisterEndpointNotificationCallback
            # 7: UnregisterEndpointNotificationCallback
            enum_endpoints_func = ctypes.WINFUNCTYPE(
                HRESULT, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, POINTER(ctypes.c_void_p)
            )(enum_vtable[3])
            get_default_func = ctypes.WINFUNCTYPE(
                HRESULT, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, POINTER(ctypes.c_void_p)
            )(enum_vtable[4])
            release_enum = ctypes.WINFUNCTYPE(c_ulong, ctypes.c_void_p)(enum_vtable[2])

            # Query Default Endpoints
            # eRender Console
            p_dev = ctypes.c_void_p()
            if get_default_func(enum_ptr, E_RENDER, E_CONSOLE, ctypes.byref(p_dev)) == 0 and p_dev:
                info = self._query_endpoint_details(p_dev, "render")
                if info:
                    inventory.default_console_render_id = info.id
                _release_com_ptr(p_dev)

            # eRender Communications
            p_dev = ctypes.c_void_p()
            if (
                get_default_func(enum_ptr, E_RENDER, E_COMMUNICATIONS, ctypes.byref(p_dev)) == 0
                and p_dev
            ):
                info = self._query_endpoint_details(p_dev, "render")
                if info:
                    inventory.default_communications_render_id = info.id
                _release_com_ptr(p_dev)

            # eCapture Console
            p_dev = ctypes.c_void_p()
            if get_default_func(enum_ptr, E_CAPTURE, E_CONSOLE, ctypes.byref(p_dev)) == 0 and p_dev:
                info = self._query_endpoint_details(p_dev, "capture")
                if info:
                    inventory.default_console_capture_id = info.id
                _release_com_ptr(p_dev)

            # eCapture Communications
            p_dev = ctypes.c_void_p()
            if (
                get_default_func(enum_ptr, E_CAPTURE, E_COMMUNICATIONS, ctypes.byref(p_dev)) == 0
                and p_dev
            ):
                info = self._query_endpoint_details(p_dev, "capture")
                if info:
                    inventory.default_communications_capture_id = info.id
                _release_com_ptr(p_dev)

            # Enumerate Render Endpoints
            col_ptr = ctypes.c_void_p()
            if (
                enum_endpoints_func(enum_ptr, E_RENDER, DEVICE_STATE_ACTIVE, ctypes.byref(col_ptr))
                == 0
                and col_ptr
            ):
                col_vtable = ctypes.cast(col_ptr, POINTER(POINTER(ctypes.c_void_p))).contents
                # IMMDeviceCollection: 0: QI, 1: AddRef, 2: Release, 3: GetCount, 4: Item
                count_func = ctypes.WINFUNCTYPE(HRESULT, ctypes.c_void_p, POINTER(wintypes.UINT))(
                    col_vtable[3]
                )
                item_func = ctypes.WINFUNCTYPE(
                    HRESULT, ctypes.c_void_p, wintypes.UINT, POINTER(ctypes.c_void_p)
                )(col_vtable[4])
                release_col = ctypes.WINFUNCTYPE(c_ulong, ctypes.c_void_p)(col_vtable[2])

                count = wintypes.UINT()
                count_func(col_ptr, ctypes.byref(count))
                for i in range(count.value):
                    d_ptr = ctypes.c_void_p()
                    if item_func(col_ptr, i, ctypes.byref(d_ptr)) == 0 and d_ptr:
                        ep = self._query_endpoint_details(d_ptr, "render")
                        if ep:
                            ep.is_default_console = ep.id == inventory.default_console_render_id
                            ep.is_default_communications = (
                                ep.id == inventory.default_communications_render_id
                            )
                            inventory.render_devices.append(ep)
                            if ep.is_vb_cable_input:
                                inventory.vb_cable_installed = True
                            if ep.is_physical and ep.form_factor in (
                                EndpointFormFactor.HEADPHONES,
                                EndpointFormFactor.HEADSET,
                                EndpointFormFactor.SPEAKERS,
                            ):
                                inventory.physical_headphones_detected = True
                        _release_com_ptr(d_ptr)
                release_col(col_ptr)

            # Enumerate Capture Endpoints
            col_ptr = ctypes.c_void_p()
            if (
                enum_endpoints_func(enum_ptr, E_CAPTURE, DEVICE_STATE_ACTIVE, ctypes.byref(col_ptr))
                == 0
                and col_ptr
            ):
                col_vtable = ctypes.cast(col_ptr, POINTER(POINTER(ctypes.c_void_p))).contents
                count_func = ctypes.WINFUNCTYPE(HRESULT, ctypes.c_void_p, POINTER(wintypes.UINT))(
                    col_vtable[3]
                )
                item_func = ctypes.WINFUNCTYPE(
                    HRESULT, ctypes.c_void_p, wintypes.UINT, POINTER(ctypes.c_void_p)
                )(col_vtable[4])
                release_col = ctypes.WINFUNCTYPE(c_ulong, ctypes.c_void_p)(col_vtable[2])

                count = wintypes.UINT()
                count_func(col_ptr, ctypes.byref(count))
                for i in range(count.value):
                    d_ptr = ctypes.c_void_p()
                    if item_func(col_ptr, i, ctypes.byref(d_ptr)) == 0 and d_ptr:
                        ep = self._query_endpoint_details(d_ptr, "capture")
                        if ep:
                            ep.is_default_console = ep.id == inventory.default_console_capture_id
                            ep.is_default_communications = (
                                ep.id == inventory.default_communications_capture_id
                            )
                            inventory.capture_devices.append(ep)
                            if ep.is_vb_cable_output:
                                inventory.vb_cable_installed = True
                            if ep.is_physical and ep.form_factor in (
                                EndpointFormFactor.MICROPHONE,
                                EndpointFormFactor.HEADSET,
                            ):
                                inventory.physical_microphone_detected = True
                        _release_com_ptr(d_ptr)
                release_col(col_ptr)

            release_enum(enum_ptr)

        except Exception as e:
            logger.error("Exception in Windows MMDevice enumeration", error=str(e))
        finally:
            if co_initialized:
                self._ole32.CoUninitialize()

        return inventory


# ---------------------------------------------------------------------------
# Fake Device Provider (For testing & non-Windows dev)
# ---------------------------------------------------------------------------
class FakeDeviceProvider:
    """Mock device provider returning a configurable device inventory."""

    def __init__(self, inventory: AudioDeviceInventory | None = None) -> None:
        if inventory is not None:
            self.inventory = inventory
        else:
            self.inventory = AudioDeviceInventory(
                capture_devices=[
                    AudioEndpointInfo(
                        id="{0.0.1.00000000}.{realtek-mic-guid}",
                        friendly_name="Microphone Array (Realtek Audio)",
                        adapter_name="Realtek High Definition Audio",
                        data_flow="capture",
                        form_factor=EndpointFormFactor.MICROPHONE,
                        is_default_console=True,
                        is_default_communications=True,
                        is_physical=True,
                    ),
                    AudioEndpointInfo(
                        id="{0.0.1.00000000}.{vbcable-out-guid}",
                        friendly_name=VB_CABLE_OUTPUT_NAME,
                        adapter_name="VB-Audio Virtual Cable",
                        data_flow="capture",
                        form_factor=EndpointFormFactor.LINE_LEVEL,
                        is_vb_cable_output=True,
                        is_physical=False,
                    ),
                ],
                render_devices=[
                    AudioEndpointInfo(
                        id="{0.0.0.00000000}.{realtek-hp-guid}",
                        friendly_name="Speakers / Headphones (Realtek Audio)",
                        adapter_name="Realtek High Definition Audio",
                        data_flow="render",
                        form_factor=EndpointFormFactor.HEADPHONES,
                        is_default_console=True,
                        is_default_communications=True,
                        is_physical=True,
                    ),
                    AudioEndpointInfo(
                        id="{0.0.0.00000000}.{vbcable-in-guid}",
                        friendly_name=VB_CABLE_INPUT_NAME,
                        adapter_name="VB-Audio Virtual Cable",
                        data_flow="render",
                        form_factor=EndpointFormFactor.LINE_LEVEL,
                        is_vb_cable_input=True,
                        is_physical=False,
                    ),
                ],
                default_console_render_id="{0.0.0.00000000}.{realtek-hp-guid}",
                default_communications_render_id="{0.0.0.00000000}.{realtek-hp-guid}",
                default_console_capture_id="{0.0.1.00000000}.{realtek-mic-guid}",
                default_communications_capture_id="{0.0.1.00000000}.{realtek-mic-guid}",
                vb_cable_installed=True,
                physical_microphone_detected=True,
                physical_headphones_detected=True,
            )

    def get_inventory(self) -> AudioDeviceInventory:
        return self.inventory


# ---------------------------------------------------------------------------
# Global Provider Dispatcher
# ---------------------------------------------------------------------------
_active_provider: AudioDeviceProvider | None = None


def get_audio_device_provider() -> AudioDeviceProvider:
    global _active_provider
    if _active_provider is None:
        if sys.platform == "win32" and os.environ.get("HTH_USE_FAKE_AUDIO_DEVICES") != "1":
            _active_provider = WindowsMMDeviceProvider()
        else:
            _active_provider = FakeDeviceProvider()
    return _active_provider


def set_audio_device_provider(provider: AudioDeviceProvider) -> None:
    """Set the active provider (used by test suites)."""
    global _active_provider
    _active_provider = provider


def list_audio_devices() -> list[DeviceInfo]:
    """Phase 2 backwards-compatibility endpoint."""
    inv = get_audio_device_provider().get_inventory()
    results: list[DeviceInfo] = []

    for d in inv.capture_devices:
        results.append(
            DeviceInfo(
                id=d.id,
                friendly_name=d.friendly_name,
                device_type=DeviceType.VIRTUAL if d.is_vb_cable_output else DeviceType.INPUT,
                sample_rates=[16000, 44100, 48000],
                default_sample_rate=48000,
                is_default=d.is_default_communications,
            )
        )

    for d in inv.render_devices:
        results.append(
            DeviceInfo(
                id=d.id,
                friendly_name=d.friendly_name,
                device_type=DeviceType.VIRTUAL if d.is_vb_cable_input else DeviceType.OUTPUT,
                sample_rates=[16000, 44100, 48000],
                default_sample_rate=48000,
                is_default=d.is_default_communications,
            )
        )

    return results


def get_device_inventory() -> AudioDeviceInventory:
    """Return the complete Phase 4 device inventory with stable GUIDs and roles."""
    return get_audio_device_provider().get_inventory()
