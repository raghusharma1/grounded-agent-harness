# Rate limiting

Rate limiting caps how many requests a client may make in a period. It protects a service from overload and abuse, and it keeps one noisy client from degrading everyone else.

## Token bucket

A token bucket holds up to a fixed number of tokens and refills at a steady rate. Each request spends one token, and a request that finds the bucket empty is rejected or delayed. Because a full bucket can be spent at once, the token bucket allows short bursts up to the bucket size while enforcing the average rate over time.

## Leaky bucket

A leaky bucket queues requests and processes them at a constant rate, like water leaking from a hole. Bursts are smoothed into a steady output, and requests that arrive when the queue is full are dropped. It suits systems that need a constant processing rate more than burst tolerance.

## Window counters

A fixed window counter counts requests per calendar window, such as each minute. It is cheap, but a client can send a full quota at the end of one window and another at the start of the next, briefly doubling the allowed rate. A sliding window log stores each request timestamp and counts those inside the last window, which is exact but uses memory per request. A sliding window counter estimates the sliding count from the current and previous fixed windows, weighted by how far into the current window we are, which is nearly as accurate at a fraction of the memory.

## Distributed limits and responses

When many servers enforce one limit, the counters usually live in a shared store such as Redis, updated atomically. A rejected request should return HTTP 429 Too Many Requests, ideally with a Retry-After header telling the client when to try again.
