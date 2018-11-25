WTF_CSRF_ENABLED = True
SECRET_KEY = 'you-will-never-guess'
SESSION_TYPE = 'filesystem'
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
