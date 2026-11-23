"""Deny Internet sockets in the diagnostic process and inherited children."""

import ctypes
import errno
import platform


def deny_internet_sockets():
    """Install a Linux seccomp filter while retaining local UNIX IPC."""
    assert platform.system() == 'Linux' and platform.machine() == 'x86_64'
    class Filter(ctypes.Structure):
        _fields_ = [('code', ctypes.c_ushort), ('jt', ctypes.c_ubyte), ('jf', ctypes.c_ubyte), ('k', ctypes.c_uint)]
    class Program(ctypes.Structure):
        _fields_ = [('len', ctypes.c_ushort), ('filter', ctypes.POINTER(Filter))]
    # seccomp_data: syscall number at 0, first 64-bit argument at 16.
    # x86_64 socket(2) is syscall 41. Deny AF_INET and AF_INET6 only.
    instructions = (Filter * 7)(
        Filter(0x20, 0, 0, 0), Filter(0x15, 0, 4, 41),
        Filter(0x20, 0, 0, 16), Filter(0x15, 1, 0, 2),
        Filter(0x15, 0, 1, 10), Filter(0x06, 0, 0, 0x00050000 | errno.EPERM),
        Filter(0x06, 0, 0, 0x7FFF0000))
    program = Program(len(instructions), instructions)
    libc = ctypes.CDLL(None, use_errno=True)
    assert libc.prctl(38, 1, 0, 0, 0) == 0, ctypes.get_errno()
    assert libc.prctl(22, 2, ctypes.byref(program), 0, 0) == 0, ctypes.get_errno()
