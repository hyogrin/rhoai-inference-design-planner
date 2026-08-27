from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_env: str = "development"
    app_log_level: str = "INFO"
    app_cors_origins: str = "http://localhost:3000"

    database_url: str = "postgresql+asyncpg://planner:planner@localhost:5432/inference_planner"
    database_url_sync: str = "postgresql://planner:planner@localhost:5432/inference_planner"

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model_name: str = "gpt-4o"

    hf_token: str = ""

    mcp_web_search_url: str = "http://127.0.0.1:9003"
    searxng_url: str = ""
    verify_ssl: bool = True

    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_tracking_token: str = ""
    mlflow_tracking_insecure_tls: bool = False
    mlflow_experiment_name: str = "inference-design-planner"
    mlflow_workspace: str = ""
    prometheus_enabled: bool = True

    anthropic_api_key: str = ""
    golden_model_name: str = "claude-opus-4-6-20250826"

    model_config = {"env_file": ".env", "extra": "ignore"}


def get_settings() -> Settings:
    return Settings()
