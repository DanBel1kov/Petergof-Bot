# Petergof-Bot

Чат-бот — ассистент по государственному музею-заповеднику "Петергоф". Ссылка: **https://t.me/peterhof_robot**

![GitHub Stars](https://img.shields.io/github/stars/AntiSlang/Petergof-Bot?style=for-the-badge)
[![Telegram Bot](https://img.shields.io/badge/🤖_Telegram_Bot-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/peterhof_robot)

## Функциональные возможности

- Регистрация, понимание работы бота: /start, /help
- Ответы на вопросы по объектам музея
- Ответы на вопросы о билетах, расписаниях, новостях
- Построение маршрутов: /route
- Поддержка пользователей: /support
- Настройки и персонализация: /settings

## Технологии

- Python, Telegram API (aiogram)
- RAG, YandexGPT‑5, RAPTOR

## Запуск проекта

1. Склонировать репозиторий:
   ```bash
   git clone https://github.com/AntiSlang/Petergof-Bot.git
   ```
2. Установить пакеты python
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Настроить переменные окружения в `.env`
4. Запустить:
   ```bash
   python bot.py
   ```

## Деплой

- Бот расположен на виртуальной машине Yandex Compute Cloud

