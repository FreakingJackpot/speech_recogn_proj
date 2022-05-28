import subprocess
import platform
from functools import partial

from dynaconf import Dynaconf
from pyautogui import press, hotkey

from config import settings

SHUTDOWN_COMMANDS = {
    'Windows': 'shutdown /s',
    'Darwin': 'shutdown -h n ow',
    'Linux': 'shutdown -h now',
}

SUSPEND_COMMANDS = {
    'Windows': 'rundll32 powrprof.dll,SetSuspendState 0,1,0',
    'Darwin': 'pmset sleepnow',
    'Linux': 'systemctl hibernate',
}

CALCULATOR_COMMANDS = {
    'Windows': 'calc',
    'Darwin': 'open -a Calculator',
    'Linux': 'gnome-calculator',
}

SCISSORS_HOTKEYS = {
    'Windows': partial(hotkey, 'win', 'shift', 's'),
    'Darwin': partial(hotkey, 'ctrlleft', 'shift', '4'),
    'Linux': partial(hotkey, 'ctrlleft', 'shift', 'printscreen'),
}

CALENDAR_HOTKEYS = {
    'Windows': partial(hotkey, 'win', 'alt', 'd'),
    'Darwin': partial(hotkey, ''),
    'Linux': partial(hotkey, 'win', 'm'),
}


class OsCommands:
    def __init__(self):
        self._platform = settings.get('platform', platform.system())
        self._volume_step = settings.get('volume_step', 20)
        self._calculator_command = settings.get('calculator_command', CALCULATOR_COMMANDS[self._platform])
        self._scissors_hotkey = self._get_scissors_hotkey(settings)
        self._calendar_hotkey = self._get_calendar_hotkey(settings)

    def _get_scissors_hotkey(self, settings_: Dynaconf) -> partial:
        scissors_hotkey_str = settings_.get('scissors_hotkey', '')
        if scissors_hotkey_str:
            keys = scissors_hotkey_str.split('+')
            return partial(hotkey, *keys)

        return SCISSORS_HOTKEYS[self._platform]

    def _get_calendar_hotkey(self, settings_: Dynaconf) -> partial:
        calendar_hotkey_str = settings_.get('calendar_hotkey', '')
        if calendar_hotkey_str:
            keys = calendar_hotkey_str.split('+')
            return partial(hotkey, *keys)

        return CALENDAR_HOTKEYS[self._platform]

    def shutdown(self) -> None:
        subprocess.run(SHUTDOWN_COMMANDS[self._platform])

    def suspend(self) -> None:
        subprocess.run(SUSPEND_COMMANDS[self._platform])

    def volume_up(self) -> None:
        press('volumeup', presses=self._volume_step // 2)

    def volume_down(self) -> None:
        press('volumedown', presses=self._volume_step // 2)

    def mute(self) -> None:
        press('volumemute')

    def unmute(self) -> None:
        press('volumemute')

    def open_calculator(self) -> None:
        subprocess.run(self._calculator_command)

    def open_scissors(self) -> None:
        print(self._scissors_hotkey)
        self._scissors_hotkey()

    def open_calendar(self) -> None:
        self._calendar_hotkey()
