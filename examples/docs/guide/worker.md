# Demo Service Guide

## Demo worker unavailable

The fictional alarm demo.svc.0001 indicates that the demo worker is not responding to the local service. This example describes an invented software service for exercising documentation retrieval and answer generation. It is not a procedure for any real product.

### Diagnosis and recovery

The demo worker can be unavailable when its local process has stopped or when the service configuration points to the wrong local endpoint. Inspect the demo worker status in the example service dashboard. Check that the configured endpoint matches the local demo worker. Restart the demo worker using the example service dashboard. Confirm that the demo worker reports a ready status before retrying the failed request.
