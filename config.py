from dynaconf import Dynaconf

settings = Dynaconf(
    settings_files=['settings.yml', 'settings.toml']
)
