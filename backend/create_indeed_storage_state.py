import os
from pathlib import Path


def main() -> None:
    """
    One-time helper:
    - Opens a Chromium browser
    - You login to Indeed manually
    - Saves Playwright storage state (cookies/session) to storage_state.json

    Usage (PowerShell):
      .\.venv\Scripts\Activate.ps1
      python -m playwright install chromium
      python backend\create_indeed_storage_state.py
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Playwright is not installed. Run:\n"
            "  pip install playwright\n"
            "  python -m playwright install chromium\n"
        ) from exc

    root = Path(__file__).resolve().parent.parent
    out_path = os.getenv("INDEED_STORAGE_STATE", str(root / "storage_state.json"))
    out_path = out_path.strip().strip('"')

    url = os.getenv("INDEED_LOGIN_URL", "https://employers.indeed.com/")

    print("\n[Indeed storage state generator]")
    print(f"- Login URL: {url}")
    print(f"- Output file: {out_path}")
    print("\nA browser will open now.")
    print("1) Login to Indeed (the same account that can view/download resumes).")
    print("2) After you see you are logged in, come back here and press ENTER.\n")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60000)

        try:
            input("Press ENTER after login is complete...")
        except KeyboardInterrupt:
            print("\nCancelled.")
            context.close()
            browser.close()
            return

        # Save cookies/session
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        context.storage_state(path=out_path)
        print(f"\nSaved: {out_path}")

        context.close()
        browser.close()


if __name__ == "__main__":
    main()

