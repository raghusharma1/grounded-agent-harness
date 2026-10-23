# Load balancing

A load balancer spreads incoming requests across several servers so that no single server becomes a bottleneck, and so that traffic can route around a server that fails.

## Layer 4 and layer 7

A layer 4 load balancer routes on network information such as IP address and TCP port. It does not look inside the request, so it is fast and protocol-agnostic. A layer 7 load balancer reads the application request, for example the HTTP path, headers or cookies, so it can route /api traffic and /static traffic to different pools, terminate TLS, or rewrite requests, at the cost of more processing per request.

## Balancing algorithms

Round robin sends each new request to the next server in turn. It is simple but assumes servers and requests are roughly equal. Weighted round robin gives larger servers a bigger share. Least connections sends the request to the server with the fewest active connections, which handles requests of uneven duration better. IP hash picks a server from a hash of the client address, so the same client keeps reaching the same server.

## Health checks and sticky sessions

Health checks probe each server periodically and remove servers that fail from rotation until they recover. Active checks send synthetic requests; passive checks watch real traffic for errors and timeouts.

Sticky sessions pin a user to one server, usually with a cookie, so in-memory session state stays available. They make scaling and failover harder, because that server's users lose their session if it goes down. Storing session state in a shared store such as Redis removes the need for stickiness.
