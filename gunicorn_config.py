workers = 1
worker_class = 'sync'
timeout = 300  # Increased from 120
keepalive = 5
max_requests = 100  # Reduced from 1000
max_requests_jitter = 10
graceful_timeout = 60
# worker_tmp_dir = '/dev/shm'  # Use RAM for temp files