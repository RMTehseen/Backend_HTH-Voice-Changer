"""Windows process management: Job Object auto-cleanup and single-instance locks."""

import ctypes
import os
import sys
from typing import Optional

# Windows Job Object API constants
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JobObjectExtendedLimitInformation = 9


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryLimit", ctypes.c_size_t),
        ("PeakJobMemoryLimit", ctypes.c_size_t),
    ]


_job_handle: Optional[int] = None


def attach_kill_on_close_job() -> bool:
    """
    On Windows, assign the current process to a Job Object configured with
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. Any child processes spawned by this
    process will also join the job, guaranteeing that if this process terminates
    or is killed, the entire process tree terminates immediately without orphans.
    """
    global _job_handle
    if sys.platform != "win32":
        return False

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return False

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

        success = kernel32.SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not success:
            kernel32.CloseHandle(job)
            return False

        current_process = kernel32.GetCurrentProcess()
        assigned = kernel32.AssignProcessToJobObject(job, current_process)
        if not assigned:
            # Could fail if already in a job that forbids breakaway; keep handle open
            pass

        _job_handle = job
        return True
    except Exception:
        return False


class SingleInstanceLock:
    """
    Cross-platform single-instance file lock placed in %LOCALAPPDATA%\\HTH\\backend.lock.
    Ensures only one backend service instance runs at a time.
    """

    def __init__(self, lock_path: str) -> None:
        self.lock_path = lock_path
        self._file_handle: Optional[int] = None
        self.is_locked = False

    def acquire(self) -> bool:
        """Attempt to acquire exclusive lock. Returns True if acquired, False if already running."""
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        try:
            if sys.platform == "win32":
                import msvcrt

                # Open with read/write access
                self._file_handle = os.open(
                    self.lock_path,
                    os.O_CREAT | os.O_RDWR,
                )
                # Non-blocking exclusive lock
                msvcrt.locking(self._file_handle, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                self._file_handle = os.open(
                    self.lock_path,
                    os.O_CREAT | os.O_RDWR,
                )
                fcntl.flock(self._file_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Write current PID
            os.ftruncate(self._file_handle, 0)
            os.write(self._file_handle, str(os.getpid()).encode("utf-8"))
            self.is_locked = True
            return True
        except (OSError, IOError):
            if self._file_handle is not None:
                try:
                    os.close(self._file_handle)
                except OSError:
                    pass
                self._file_handle = None
            return False

    def release(self) -> None:
        """Release the file lock."""
        if not self.is_locked or self._file_handle is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                try:
                    msvcrt.locking(self._file_handle, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl

                try:
                    fcntl.flock(self._file_handle, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(self._file_handle)
            if os.path.exists(self.lock_path):
                try:
                    os.remove(self.lock_path)
                except OSError:
                    pass
        finally:
            self._file_handle = None
            self.is_locked = False
