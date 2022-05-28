# MVP Проекта "Голосовой асистент" команды "Помойка"

За основу взята модель Wave2Vec с huggingface

## Команда:

* Абрамкина карина
* Абросимов Алексей
* Бирюкова Анастасия
* Волкова Анастасия
* Сергеев Сергей

## Quick start

### Install

`pip install -r requirements.txt`  
`pipwin install pyaudio`

### Запуск

`python main.py`

## Настройки

Для настройки конфигурации используется dynaconf, но можно и без них

### Параметры конфигурации

* platform: Операционка - принимает 3 параметра (Windows,Darwin,Linux)
* audio_length: Длина записываемой дорожки в секундах. default - 3
* repository_name: Имя репозитория huggingface
* browser: Название браузера. huggingface
* browser_path: Путь до исполняемого файла браузера
* home_url: Домашняя страница браузера. default - [google](https://google.com)
* currency_url: Страница курса валют. default - [banki.ru](https://www.banki.ru/products/currency/cb/)
* volume_step: Шаг увеличения/уменьшения звука. default - 20
* calculator_command: Команда запускающая колькулятор. default - системные (Windows,Ubuntu,Mac)
* scissors_hotkey: Хоткей ножниц вида ctrl+win. default - системные
* calendar_hotkey: Хоткей календаря вида ctrl+win. default - системные, кроме mac

ЕСЛИ ШО, ТО НА МАКЕ И ЛИНУКСЕ НЕ ТЕСТИЛ 


