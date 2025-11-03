from collections import OrderedDict
import logging


logger = logging.getLogger(__name__)


class LRUCache:
    """A small process-local LRU cache for loaded files.

    This is intentionally simple and lives at module scope so it is
    safe with different multiprocessing start methods (spawn/pickling).
    The loader callable should accept a single key (file path) and return the value.
    """

    def __init__(self, maxsize: int, loader):
        self.maxsize = max(1, int(maxsize))
        self.loader = loader
        self._cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        # Cache hit
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            logger.debug(f"LRU hit: {key} (size={len(self._cache)}/{self.maxsize})")
            return self._cache[key]

        # Miss -> load
        self.misses += 1
        logger.debug(f"LRU miss: {key} (loading) -> (size_before={len(self._cache)})")
        value = self.loader(key)

        # Insert and evict if needed
        self._cache[key] = value
        if len(self._cache) > self.maxsize:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.debug(f"LRU evict: {evicted_key} (new_size={len(self._cache)})")

        return value

    def info(self):
        return {
            "maxsize": self.maxsize,
            "current_size": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
        }
