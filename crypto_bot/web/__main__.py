from __future__ import annotations

import os

import uvicorn
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    # Varsayılan yalnızca bu makineden erişim; ağa açmak için DASHBOARD_HOST=0.0.0.0 + DASHBOARD_TOKEN
    host = os.getenv("DASHBOARD_HOST", "127.0.0.1")
    port = int(os.getenv("DASHBOARD_PORT", "8501"))
    uvicorn.run("crypto_bot.web.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
