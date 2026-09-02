from yolov4_detection import resolve_providers

CPU = "CPUExecutionProvider"
COREML = "CoreMLExecutionProvider"
OPENVINO = "OpenVINOExecutionProvider"


def test_requested_providers_win_in_order():
    assert resolve_providers([COREML, CPU], [CPU, COREML]) == [COREML, CPU]


def test_unavailable_requested_providers_are_dropped():
    """A config written for the Mac should still start on a plain CPU host."""
    assert resolve_providers([COREML, CPU], [CPU]) == [CPU]


def test_defaults_prefer_coreml_when_available():
    assert resolve_providers([], [CPU, COREML]) == [COREML, CPU]


def test_defaults_fall_back_to_cpu():
    assert resolve_providers([], [CPU]) == [CPU]


def test_never_returns_empty():
    """Even a fully unsatisfiable request has to yield something runnable."""
    assert resolve_providers([OPENVINO], [CPU]) == [CPU]
    assert resolve_providers([], []) == [CPU]
