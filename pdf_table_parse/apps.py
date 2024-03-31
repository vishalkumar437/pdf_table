import asyncio
import os
import signal
import requests
from django.apps import AppConfig


def fire_and_forget(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, *kwargs)

    return wrapped


class TrainerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'pdf_table_parse'

    @fire_and_forget
    def execute_with_env(self, json):
        print(f'Sending {json} to localhost')
        try:
            requests.post('http://localhost:8080/', json=json)
        finally:
            os.system("pkill gunicorn")
            signal.raise_signal(signal.SIGTERM)

    def ready(self):
        asyncio.ensure_future(self.execute_with_env())
