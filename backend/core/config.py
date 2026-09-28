from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- LLM ----
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_JUDGE_MODEL: str = "deepseek-chat"
    #: Step 4 的总结模型单独一个：它要「看图说话」。我们自己的 WL 图像块没有任何
    #: MCP 侧标注（raw block 只有 data + mimeType），只能把图交给多模态模型自己认。
    #: deepseek-flash = DeepSeek-V4.1-Flash，input_modalities 含 image。
    DEEPSEEK_SUMMARY_MODEL: str = "deepseek-flash"

    # ---- 服务 ----
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ---- 采样 ----
    ACT_TEMPERATURE: float = 0.7
    GROUP_TEMPERATURE: float = 1.0
    MAX_CONCURRENCY: int = 8

    # ---- 训练超参 ----
    GRPO_GROUP_SIZE: int = 4
    MAX_STEPS: int = 3
    MAX_ROLLOUTS: int = 24
    ADVANTAGE_EPS: float = 1e-4

    # ---- 奖励构成：Wolfram 资源利用率占多数（用户口径）----
    #: 利用率 = 四信号等权均值，见 judge._utilization
    UTIL_WEIGHT: float = 0.55
    VERIFY_WEIGHT: float = 0.30
    JUDGE_WEIGHT: float = 0.15

    # ---- 策略资产 θ 的边界 ----
    MAX_RULES: int = 24
    MIN_RULE_GAIN: float = 0.05

    # ---- 环境 ----
    ASSET_DIR: str = "assets"
    #: 传给 WolframLanguageEvaluator 的 timeConstraint（秒），由服务端兜底限长
    EXEC_TIMEOUT: float = 30.0
    MAX_CODE_CHARS: int = 20000

    # ---- 安全红线 ----
    SAFETY_ENFORCE: bool = True

    # ---- 可审计轨迹 ----
    TRACE_DIR: str = "runs"
    TRACE_ENABLED: bool = True

    # ---- 自治边界 ----
    TOKEN_BUDGET: int = 0
    MAX_FLAT_GROUPS: int = 4
    ROLLBACK_ON_REGRESSION: bool = True

    # ---- 第二层：Wolfram 官方 MCP（唯一执行通道）----
    #: Wolfram 托管 MCP，实测无需鉴权
    MCP_URL: str = "https://agenttools.wolfram.com/mcp"
    #: MCP 请求超时（秒）
    MCP_TIMEOUT: float = 60.0
    #: 单条结果回灌给模型做自然语言总结时的字符上限
    RESULT_CHARS: int = 4000
    #: 单次求值最多带回的图像张数（一张约 14KB base64，不设上限会撑大响应体）
    MAX_IMAGES: int = 4

    # ---- 文档检索 ----
    #: agent 调用 wolfram_context 时回灌给模型的文档字符上限
    DOC_CHARS: int = 3000

    # ---- 会话记忆 ----
    #: 保留最近 N 轮对话用于指代消解（指令 3）
    MEMORY_ROUNDS: int = 3

    @property
    def llm_configured(self) -> bool:
        return bool(self.DEEPSEEK_API_KEY.strip())


settings = Settings()
