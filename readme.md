# PW2 Timelapse Exporter (GUI)

Turns **PowerWash Simulator 2** `timelapse.sav` files into a video or GIF.
Pick your SaveData folder, choose the level, hit **Start Export**. That’s it.

- Finds the Steam SaveData folder automatically (Windows).
- Lists only levels that actually have a `timelapse.sav`.
- Works with spaces and non-ASCII paths (Cyrillic etc.)
- No admin rights needed.

---

## Quick start (run from source)

**Requirements**
- Windows 10/11, 64-bit
- Python 3.9+ from [python.org](https://www.python.org/downloads/) (check “Add to PATH”)

**Install**
```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
