from src.health_monitor import HealthMonitor

def test_httpx_400_bad_request_is_neutral():
    err = "HTTPStatusError:Client error '400 Bad Request' for url 'http://192.168.206.2:8106/v1/embeddings'"
    assert HealthMonitor._failure_kind(err) == "neutral"

def test_httpx_404_is_neutral():
    err = "HTTPStatusError:Client error '404 Not Found' for url 'http://x/v1/embeddings'"
    assert HealthMonitor._failure_kind(err) == "neutral"

def test_connect_error_is_endpoint():
    assert HealthMonitor._failure_kind("ConnectError:All connection attempts failed") == "endpoint"
