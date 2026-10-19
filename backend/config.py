from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_env: str = "development"
    app_log_level: str = "INFO"
    app_cors_origins: str = "http://localhost:3000"

    database_url: str = "postgresql+asyncpg://planner:planner@localhost:5432/inference_planner"
    database_url_sync: str = "postgresql://planner:planner@localhost:5432/inference_planner"

    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o"

    hf_token: str = ""

    verify_ssl: bool = True

    mlflow_tracking_uri: str = ""
    mlflow_tracking_token: str = ""
    mlflow_tracking_auth: str = ""
    mlflow_tracking_insecure_tls: bool = False
    mlflow_experiment_name: str = "inference-design-planner"
    mlflow_workspace: str = ""
    prometheus_enabled: bool = True

    anthropic_api_key: str = ""
    golden_model_name: str = "claude-opus-4-6-20250826"

    model_config = {"env_file": ".env", "extra": "ignore"}


def get_settings() -> Settings:
    return Settings()
