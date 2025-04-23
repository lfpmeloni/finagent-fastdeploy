import logging
import os

def track_event_if_configured(event_name: str, event_data: dict):
    instrumentation_key = os.getenv("APPLICATIONINSIGHTS_INSTRUMENTATION_KEY")
    if not instrumentation_key:
        logging.warning(f"Skipping track_event for {event_name} as Application Insights is not configured")
        return

    try:
        # import inside function so missing package doesn't break startup
        from azure.monitor.events.extension import track_event
    except ImportError:
        logging.warning("azure-monitor-events-extension not installed; cannot send telemetry")
        return

    try:
        track_event(event_name, event_data)
    except Exception as ex:
        logging.error(f"Failed to send telemetry for {event_name}: {ex}")
