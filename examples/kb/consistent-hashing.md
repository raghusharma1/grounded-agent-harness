# Consistent hashing

Consistent hashing assigns keys to servers so that adding or removing a server moves only a small fraction of the keys.

## Why modulo hashing breaks

The simple approach maps a key to server number hash(key) mod N. When N changes, almost every key maps to a different server, so a cache cluster loses nearly all its hits at once and a storage cluster has to move most of its data.

## The hash ring

Consistent hashing places both servers and keys on the same circular hash space, often called a ring. A key belongs to the first server found by walking clockwise from the key's position. When a server is added, it only takes over keys between itself and its predecessor; when a server is removed, only its keys move, to the next server clockwise. On average only about K/N keys move, where K is the number of keys and N the number of servers.

## Virtual nodes

With few servers, positions on the ring are uneven and some servers own far more keys than others. Virtual nodes fix this by placing each physical server at many points on the ring. Load spreads more evenly, a failed server's keys scatter across many survivors instead of one, and a more powerful server can be given more virtual nodes so it takes a larger share.

## Where it is used

Distributed caches, partitioned databases such as Cassandra and Dynamo-style stores, and some load balancers use consistent hashing to decide which node owns which key.
