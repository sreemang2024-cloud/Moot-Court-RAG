from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: str = "data"
    raw_docs_dir: str = "data/raw_docs"
    processed_dir: str = "data/processed"
    index_dir: str = "data/index"

    chunk_size: int = 700
    chunk_overlap: int = 120
    top_k: int = 6
    min_context_score: float = 0.27

    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    use_openai: bool = False
    openai_api_key: str = ""
    llm_model: str = "gpt-4.1-mini"
    temperature: float = 0.0

    local_llm_model: str = "google/flan-t5-small"
    local_max_new_tokens: int = 220

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def raw_docs_path(self) -> Path:
        return Path(self.raw_docs_dir)

    @property
    def processed_path(self) -> Path:
        return Path(self.processed_dir)

    @property
    def index_path(self) -> Path:
        return Path(self.index_dir)


settings = Settings()
