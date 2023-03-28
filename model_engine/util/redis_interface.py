from .log import logger
import redis

DEFUALT_CONNECTION = redis.Redis()

TYPES = {
    b'list': list,
    b'string': str,
    b'set': set
}

def _redis_get(key, conn=DEFUALT_CONNECTION):
    t = conn.type(key)
    logger.debug(f'Key [{key}] is type: {t}')
    value = None
    if t == b'list':
        value = [x.decode('utf-8') for x in conn.lrange(key,0,-1)]
    elif t == b'string':
        value = conn.get(key).decode("utf-8")
    elif t == b'set':
        value = {x.decode('utf-8') for x in conn.smembers(key)}
    logger.info(f'Value for key [{key}]: {value} ({type(value), t})')
    return value, t


def redis_get(key, conn=DEFUALT_CONNECTION):
    return _redis_get(key, conn=conn)[0]


def redis_set(key, value, conn=DEFUALT_CONNECTION):
    logger.info(f'Setting value for [{key}] to : {value} ({type(value)})')
    if conn.exists(key):
        conn.delete(key)
    if isinstance(value, str):
        conn.set(key, value)
    elif isinstance(value, list):
        conn.rpush(key, *value)
    elif isinstance(value, set):
        conn.sadd(key, *value)
    else:
        logger.error('could not determine proper type for setting value on remote server...')

def redis_delete(key, conn=DEFUALT_CONNECTION):
    conn.delete(key)

if __name__ == '__main__':
    redis_set('foo', ['bar','baz'])
    foo = redis_get('foo')
    print(redis_get('eek'))
    print(foo)
