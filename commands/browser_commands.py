import webbrowser

from config import settings
from commands.base_commands import BaseCommands

DEFAULT_HOME_URL = 'https://google.com'
DEFAULT_CURRENCY_URL = 'https://www.banki.ru/products/currency/cb/'


class BrowserCommands(BaseCommands):
    def __init__(self):
        browser = settings.get('browser')
        browser_path = settings.get('browser_path')
        if browser_path:
            webbrowser.register(browser, None, webbrowser.BackgroundBrowser(browser_path))
        self._browser = webbrowser.get(browser)

        self._home_url = settings.get('home_url', DEFAULT_HOME_URL)
        self._currency_url = settings.get('currency_url', DEFAULT_CURRENCY_URL)

        self._command_dict = {
            'открыть браузер': self.open_browser,
            'курс валют': self.open_currency_page,
        }

    def open_browser(self) -> None:
        self._browser.open_new(self._home_url)

    def open_currency_page(self) -> None:
        self._browser.open(self._currency_url)
