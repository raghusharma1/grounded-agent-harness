# Message queues

A message queue lets one service hand work to another without waiting for it to finish. The producer writes a message and moves on; consumers read messages when they are ready. This decouples the two sides, absorbs traffic spikes, and lets consumers scale independently.

## Delivery guarantees

At-most-once delivery may lose a message but never delivers it twice. At-least-once delivery retries until the consumer acknowledges, so nothing is lost, but a message can arrive more than once. Exactly-once delivery is hard to achieve end to end; in practice systems combine at-least-once delivery with idempotent consumers, which produce the same result however many times they process a message, for example by recording processed message ids.

## Queues versus logs

In a classic queue such as RabbitMQ or Amazon SQS, a message is removed once a consumer acknowledges it, and each message goes to one consumer. In a log such as Apache Kafka, messages are appended to partitions and kept for a retention period; each consumer group tracks its own offset, so many independent consumers can read the same stream and replay it from an earlier point.

## Ordering and failures

Kafka preserves order only within a partition, so related messages are routed to the same partition by key. A message that keeps failing should not block the queue forever: after a set number of retries it moves to a dead-letter queue, where it can be inspected and replayed once the problem is fixed. Back-pressure, slowing producers when consumers fall behind, keeps queues from growing without limit.
