# Caching

A cache keeps a copy of data somewhere faster to reach than its source, so repeated reads skip the slow path. It trades memory and staleness for latency and load reduction on the backing store.

## Read and write patterns

Cache-aside (lazy loading) is the most common pattern: the application checks the cache first, and on a miss it reads from the database, then writes the result into the cache. Only data that is actually requested gets cached, but the first request for each key is slow.

Write-through writes to the cache and the database in the same operation, so the cache never serves data older than the database. Writes become slower because both stores must succeed.

Write-back (write-behind) acknowledges a write once the cache has it and flushes to the database later. Writes are fast, but data can be lost if the cache node fails before the flush.

## Eviction policies

When the cache is full, an eviction policy decides what to drop. LRU (least recently used) evicts the item that has gone longest without being accessed, which suits workloads where recent data is likely to be read again. LFU (least frequently used) evicts the item with the fewest accesses, which protects popular items from a burst of one-off reads. FIFO evicts in insertion order and ignores access patterns entirely.

## Invalidation and expiry

A time-to-live (TTL) expires an entry after a fixed period, bounding how stale it can get. Explicit invalidation deletes or updates the entry when the source changes. Deleting is usually safer than updating, because two concurrent updates can leave the cache holding the older value.

A cache stampede happens when a popular key expires and many requests miss at once and all hit the database. Common defences are request coalescing (one request rebuilds the entry while the others wait), adding random jitter to TTLs, and refreshing hot keys before they expire.
