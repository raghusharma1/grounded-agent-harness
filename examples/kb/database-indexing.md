# Database indexing

An index is an extra data structure that lets the database find rows without scanning the whole table. Reads get faster; writes get slower, because every insert, update or delete must also update each affected index, and the index takes storage.

## B-tree indexes

Most relational databases use B-tree indexes by default. A B-tree keeps keys sorted in a balanced tree of wide pages, so a lookup touches only a few pages even for very large tables. Because keys are sorted, B-trees support equality lookups, range queries such as BETWEEN or greater-than, and ORDER BY on the indexed columns.

## Hash indexes

A hash index maps each key through a hash function to a bucket. Equality lookups are very fast, but hash indexes cannot answer range queries or return rows in sorted order, because hashing destroys key order.

## Composite indexes and the leftmost prefix

A composite index covers several columns in a fixed order, for example (last_name, first_name). The database can use it for queries that filter on a leftmost prefix of those columns: last_name alone, or last_name and first_name together. A query that filters only on first_name generally cannot use it efficiently.

## Covering indexes and selectivity

A covering index contains every column a query needs, so the database can answer from the index alone without reading the table rows. An index helps most on selective columns, where a value matches a small fraction of rows; on a low-selectivity column such as a boolean flag, a full scan is often just as fast.
