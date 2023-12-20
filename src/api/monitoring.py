import os
from typing import Callable

from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_fastapi_instrumentator.metrics import Info

import logging

NAMESPACE = os.environ.get("METRICS_NAMESPACE", "fastapi")
SUBSYSTEM = os.environ.get("METRICS_SUBSYSTEM", "model")

instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
    inprogress_name="fastapi_inprogress",
    inprogress_labels=True
)

# Metrics

instrumentator.add(
    metrics.request_size(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_name="image_upload_size",
        metric_doc="Size of uploaded images",
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
instrumentator.add(
    metrics.response_size(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_name="image_download_size",
        metric_doc="Size of downloaded images",
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
instrumentator.add(
    metrics.latency(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_name="segmentation_latency",
        metric_doc="Latency of image segmentation",
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
instrumentator.add(
    metrics.requests(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_name="total_requests",
        metric_doc="Total number of requests",
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)

def segmentation_result_metric(
    metric_name: str = "segmentation_result",
    metric_doc: str = "Outcome of image segmentation",
    metric_namespace: str = "",
    metric_subsystem: str = "",
    buckets=(0, 1),
) -> Callable[[Info], None]:
    METRIC = Histogram(
        metric_name,
        metric_doc,
        labelnames=["result_type"],
        buckets=buckets,
        namespace=metric_namespace,
        subsystem=metric_subsystem,
    )
    METRIC.labels("success")
    METRIC.labels("failure")

    def instrumentation(info: Info) -> None:
        if info.modified_handler == "/predict":
            result_type = info.response.headers.get("X-image-segmentation-result")
            if result_type:
                METRIC.labels(result_type).observe(1.0)

    return instrumentation

instrumentator.add(segmentation_result_metric(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM))
