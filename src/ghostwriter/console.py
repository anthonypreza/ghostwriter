"""Shared terminal color support for PowerShell and other consoles."""

import os


def use_color(mode, terminal):
    if mode == "never":
        return False
    if mode == "auto" and ("NO_COLOR" in os.environ or not terminal.isatty()):
        return False
    if os.name == "nt":
        # Enable ANSI colors in the Windows console used by PowerShell.
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetStdHandle.argtypes = [wintypes.DWORD]
        kernel.GetStdHandle.restype = wintypes.HANDLE
        kernel.GetConsoleMode.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        handle = kernel.GetStdHandle(-11 & 0xFFFFFFFF)
        flags = wintypes.DWORD()
        enabled = kernel.GetConsoleMode(
            handle, ctypes.byref(flags)
        ) and kernel.SetConsoleMode(handle, flags.value | 4)
        return bool(enabled) or mode == "always"
    return True
