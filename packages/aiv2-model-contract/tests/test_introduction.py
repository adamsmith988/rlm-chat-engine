from aiv2_model_contract.introduction import (
    build_introduction,
    effective_serve,
    normalize_introduction,
    project_introduction_for_catalog,
)


def test_normalize_legacy_flat_z_image():
    doc = normalize_introduction(
        {
            "logical_id": "z-image-turbo",
            "pipeline_tag": "text-to-image",
            "library_name": "diffusers",
            "api_protocol": "openai_compat_images",
            "runtime": "vllm-omni",
            "input_modalities": ["text"],
            "output_modalities": ["image"],
            "pipeline_class": "ZImagePipeline",
            "run_schema": {
                "api_protocol": "openai_compat_images",
                "ui_profile": "image_gen",
                "fields": [],
            },
        }
    )
    assert doc["schema_version"] == 2
    assert doc["identity"]["logical_id"] == "z-image-turbo"
    assert doc["hub"]["library_name"] == "diffusers"
    assert doc["serve"]["api_protocol"] == "openai_compat_images"
    assert "dispatch_key" not in doc["serve"]
    assert doc["artifacts"]["diffusers"]["pipeline_class"] == "ZImagePipeline"


def test_build_and_catalog_projection():
    intro = build_introduction(
        identity={"logical_id": "ornith-1-0-9b"},
        serve={
            "api_protocol": "openai_compat_chat",
            "runtime": "vllm-openai",
            "endpoint_path": "/v1/chat/completions",
            "request_shape": "messages",
        },
        io_profile={"input_modalities": ["text"], "output_modalities": ["text"]},
    )
    proj = project_introduction_for_catalog(intro)
    assert proj["logical_id"] == "ornith-1-0-9b"
    assert proj["api_protocol"] == "openai_compat_chat"
    assert proj["endpoint_path"] == "/v1/chat/completions"
    assert proj["request_shape"] == "messages"
    assert proj["input_modalities"] == ["text"]
    assert proj["output_modalities"] == ["text"]
    assert "dispatch_key" not in proj
    assert "introduction" in proj


def test_normalize_preserves_request_shape_from_flat():
    doc = normalize_introduction(
        {
            "logical_id": "qwen-chat",
            "api_protocol": "openai_compat_chat",
            "runtime": "vllm",
            "request_shape": "state_plus_typed_questions",
            "input_modalities": ["text"],
            "output_modalities": ["text"],
        }
    )
    assert doc["serve"]["request_shape"] == "state_plus_typed_questions"
    serve = effective_serve(
        {
            "request_shape": "messages",
            "introduction": doc,
        }
    )
    assert serve["request_shape"] == "state_plus_typed_questions"


def test_effective_serve_nested_wins_over_flat():
    serve = effective_serve(
        {
            "api_protocol": "openai_compat_chat",
            "endpoint_path": "/v1/chat/completions",
            "runtime": "vllm",
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "introduction": {
                "serve": {
                    "api_protocol": "openai_compat_images",
                    "endpoint_path": "/v1/images/generations",
                    "runtime": "vllm-omni",
                    "ui_profile": "image_gen",
                    "admission": {"pool": "vllm_omni", "max_concurrent": 4},
                },
                "io": {
                    "input_modalities": ["text"],
                    "output_modalities": ["image"],
                },
            },
        }
    )
    assert serve["api_protocol"] == "openai_compat_images"
    assert serve["endpoint_path"] == "/v1/images/generations"
    assert serve["runtime"] == "vllm-omni"
    assert serve["ui_profile"] == "image_gen"
    assert serve["admission"]["pool"] == "vllm_omni"
    assert serve["input_modalities"] == ["text"]
    assert serve["output_modalities"] == ["image"]


def test_effective_serve_flat_only():
    serve = effective_serve(
        {
            "api_protocol": "openai_compat_chat",
            "endpoint_path": "/v1/chat/completions",
            "input_modalities": ["text"],
            "output_modalities": ["text"],
        }
    )
    assert serve["api_protocol"] == "openai_compat_chat"
    assert serve["endpoint_path"] == "/v1/chat/completions"
    assert serve["input_modalities"] == ["text"]
    assert serve["output_modalities"] == ["text"]


def test_effective_serve_falls_back_to_run_schema_endpoint_path():
    serve = effective_serve(
        {
            "api_protocol": "openai_compat_custom",
            "api": "audio",
            "input_modalities": ["audio"],
            "output_modalities": ["text"],
            "run_schema": {
                "family": "whisper",
                "endpoint_path": "/v1/audio/transcriptions",
            },
        }
    )
    assert serve["endpoint_path"] == "/v1/audio/transcriptions"
    assert serve["api_protocol"] == "openai_compat_custom"
