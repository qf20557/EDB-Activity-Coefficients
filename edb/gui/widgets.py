"""Shared Streamlit helpers: native OS file dialogs (Win32 ctypes, no Tk), Plotly box parsing, compound matching."""
import ctypes
from ctypes import wintypes
import math
from pathlib import Path
import sys

OFN_OVERWRITEPROMPT = 0x00000002
OFN_NOCHANGEDIR = 0x00000008
OFN_ALLOWMULTISELECT = 0x00000200
OFN_PATHMUSTEXIST = 0x00000800
OFN_FILEMUSTEXIST = 0x00001000
OFN_EXPLORER = 0x00080000

WS_POPUP = 0x80000000
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
HWND_TOPMOST = wintypes.HWND(-1)
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_SHOWWINDOW = 0x0040


class _OPENFILENAMEW(ctypes.Structure):
    _fields_ = [
        ("lStructSize", wintypes.DWORD),
        ("hwndOwner", wintypes.HWND),
        ("hInstance", wintypes.HINSTANCE),
        ("lpstrFilter", wintypes.LPCWSTR),
        ("lpstrCustomFilter", wintypes.LPWSTR),
        ("nMaxCustFilter", wintypes.DWORD),
        ("nFilterIndex", wintypes.DWORD),
        ("lpstrFile", wintypes.LPWSTR),
        ("nMaxFile", wintypes.DWORD),
        ("lpstrFileTitle", wintypes.LPWSTR),
        ("nMaxFileTitle", wintypes.DWORD),
        ("lpstrInitialDir", wintypes.LPCWSTR),
        ("lpstrTitle", wintypes.LPCWSTR),
        ("Flags", wintypes.DWORD),
        ("nFileOffset", wintypes.WORD),
        ("nFileExtension", wintypes.WORD),
        ("lpstrDefExt", wintypes.LPCWSTR),
        ("lCustData", wintypes.LPARAM),
        ("lpfnHook", wintypes.LPVOID),
        ("lpTemplateName", wintypes.LPCWSTR),
        ("pvReserved", wintypes.LPVOID),
        ("dwReserved", wintypes.DWORD),
        ("FlagsEx", wintypes.DWORD),
    ]


def _win32_apis():
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    comdlg32 = ctypes.WinDLL("comdlg32", use_last_error=True)

    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
    ]
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    user32.DestroyWindow.restype = wintypes.BOOL
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.SetForegroundWindow.restype = wintypes.BOOL
    user32.SetWindowPos.argtypes = [
        wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT,
    ]
    user32.SetWindowPos.restype = wintypes.BOOL

    comdlg32.GetOpenFileNameW.argtypes = [ctypes.POINTER(_OPENFILENAMEW)]
    comdlg32.GetOpenFileNameW.restype = wintypes.BOOL
    comdlg32.GetSaveFileNameW.argtypes = [ctypes.POINTER(_OPENFILENAMEW)]
    comdlg32.GetSaveFileNameW.restype = wintypes.BOOL
    return user32, comdlg32


def _build_filter_str(filetypes):
    """Build double-NUL-terminated Win32 filter string from [(label, '*.a;*.b'), ...]."""
    chunks = []
    for desc, pattern in filetypes:
        win_pat = ";".join(p.strip() for p in pattern.replace(",", " ").split() if p.strip())
        chunks.append(f"{desc}\0{win_pat}")
    return "\0".join(chunks) + "\0\0"


def ask_open_files(title="Select file(s)", filetypes=(("All files", "*.*"),), multiple=False, initial_dir=None):
    """Open a native Windows File Explorer open-file dialog on top of the browser.
    Returns a list of selected file paths (if multiple=True) or a single path string / None."""
    if sys.platform != "win32":
        return [] if multiple else None

    user32, comdlg32 = _win32_apis()
    hwnd = user32.CreateWindowExW(
        WS_EX_TOPMOST | WS_EX_TOOLWINDOW, "STATIC", "", WS_POPUP, 120, 120, 1, 1, None, None, None, None
    )
    try:
        if hwnd:
            user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
            user32.SetForegroundWindow(hwnd)

        buf_len = 65536 if multiple else 4096
        buf = ctypes.create_unicode_buffer(buf_len)
        flt = _build_filter_str(filetypes)

        ofn = _OPENFILENAMEW()
        ofn.lStructSize = ctypes.sizeof(_OPENFILENAMEW)
        ofn.hwndOwner = hwnd
        ofn.lpstrFilter = flt
        ofn.nFilterIndex = 1
        ofn.lpstrFile = ctypes.cast(buf, wintypes.LPWSTR)
        ofn.nMaxFile = buf_len
        ofn.lpstrInitialDir = str(initial_dir) if initial_dir else None
        ofn.lpstrTitle = title
        flags = OFN_EXPLORER | OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR
        if multiple:
            flags |= OFN_ALLOWMULTISELECT
        ofn.Flags = flags

        if not comdlg32.GetOpenFileNameW(ctypes.byref(ofn)):
            return [] if multiple else None

        if not multiple:
            return buf.value or None

        raw = ctypes.wstring_at(ctypes.addressof(buf), buf_len)
        end = raw.find("\0\0")
        if end != -1:
            raw = raw[:end]
        parts = [p for p in raw.split("\0") if p]
        if not parts:
            return []
        if len(parts) == 1:
            return [parts[0]]
        folder = Path(parts[0])
        return [str(folder / name) for name in parts[1:]]
    finally:
        if hwnd:
            user32.DestroyWindow(hwnd)


def ask_save_file(title="Save file", filetypes=(("CSV (*.csv)", "*.csv"),), initial_file="", default_ext=".csv",
                  initial_dir=None):
    """Open a native Windows File Explorer Save-As dialog on top of the browser.
    Returns the chosen file path string (with appropriate extension applied) or None if cancelled."""
    if sys.platform != "win32":
        return None

    user32, comdlg32 = _win32_apis()
    hwnd = user32.CreateWindowExW(
        WS_EX_TOPMOST | WS_EX_TOOLWINDOW, "STATIC", "", WS_POPUP, 120, 120, 1, 1, None, None, None, None
    )
    try:
        if hwnd:
            user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
            user32.SetForegroundWindow(hwnd)

        buf_len = 4096
        buf = ctypes.create_unicode_buffer(initial_file, buf_len)
        flt = _build_filter_str(filetypes)

        ofn = _OPENFILENAMEW()
        ofn.lStructSize = ctypes.sizeof(_OPENFILENAMEW)
        ofn.hwndOwner = hwnd
        ofn.lpstrFilter = flt
        ofn.nFilterIndex = 1
        ofn.lpstrFile = ctypes.cast(buf, wintypes.LPWSTR)
        ofn.nMaxFile = buf_len
        ofn.lpstrInitialDir = str(initial_dir) if initial_dir else None
        ofn.lpstrTitle = title
        ofn.lpstrDefExt = default_ext.lstrip(".") if default_ext else None
        ofn.Flags = OFN_EXPLORER | OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR

        if not comdlg32.GetSaveFileNameW(ctypes.byref(ofn)):
            return None

        chosen = Path(buf.value)
        # Apply the extension of the selected filter dropdown item if none or mismatched with filter
        idx = int(ofn.nFilterIndex) - 1
        if 0 <= idx < len(filetypes):
            patterns = [p.strip() for p in filetypes[idx][1].replace(";", " ").split() if p.strip().startswith("*.")]
            exts = [p[1:].lower() for p in patterns if p != "*.*"]
            if exts and chosen.suffix.lower() not in exts:
                chosen = chosen.with_suffix(exts[0])
        elif default_ext and not chosen.suffix:
            chosen = chosen.with_suffix(default_ext if default_ext.startswith(".") else f".{default_ext}")
        return str(chosen)
    finally:
        if hwnd:
            user32.DestroyWindow(hwnd)


def match_compound(compounds, preferred):
    """Find `preferred` in `compounds` (exact first, then case-insensitive), else return the first compound name."""
    if not compounds:
        return ""
    if preferred in compounds:
        return preferred
    target = preferred.strip().lower()
    for name in compounds:
        if name.strip().lower() == target:
            return name
    return next(iter(compounds))


def extract_selection_box(state):
    """Return (xmin, xmax, ymin, ymax) from a Streamlit Plotly selection state dict, or None."""
    if not state:
        return None
    sel = state.get("selection") if isinstance(state, dict) else getattr(state, "selection", None)
    if not sel:
        return None
    boxes = sel.get("box") if isinstance(sel, dict) else getattr(sel, "box", None)
    if not boxes:
        return None
    box = boxes[-1]
    xs = box.get("x", [])
    ys = box.get("y", [])
    if len(xs) < 2 or len(ys) < 2:
        return None
    try:
        x0, x1 = sorted((float(xs[0]), float(xs[1])))
        y0, y1 = sorted((float(ys[0]), float(ys[1])))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x0, x1, y0, y1)):
        return None
    return x0, x1, y0, y1
