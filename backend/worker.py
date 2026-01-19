import os
import redis
from rq import Queue
from rq.worker import SimpleWorker # Import SimpleWorker explicitly

listen = ['default']
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')

conn = redis.from_url(redis_url)

if __name__ == '__main__':
    print("👷 Worker started (Windows Compatibility Mode). Listening for TTS jobs...")
    
    q = Queue('default', connection=conn)
    
    # We use SimpleWorker because Windows does not support os.fork()
    worker = SimpleWorker([q], connection=conn)
    
    worker.work()