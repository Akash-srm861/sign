# Gunicorn configuration for Render deployment

import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'sync'
worker_connections = 1000
timeout = 120
keepalive = 5

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Logging
errorlog = '-'
loglevel = 'info'
accesslog = '-'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = 'sign-language-app'

# Server hooks
def on_starting(server):
    print("Starting Sign Language Learning App")

def on_reload(server):
    print("Reloading Sign Language Learning App")

def when_ready(server):
    print("Server is ready. Spawning workers")

def on_exit(server):
    print("Shutting down Sign Language Learning App")
