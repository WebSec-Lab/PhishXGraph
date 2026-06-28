#!/usr/bin/env python3
"""Capture website screenshots using headless Chrome (Selenium)."""
import os
import time


def capture_screenshot(url, output_path, timeout=30):
    """Capture a screenshot of a URL using headless Chrome.

    Uses Selenium 4's built-in driver manager to automatically resolve
    Chrome/chromedriver version mismatches.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--hide-scrollbars")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        for chrome_path in [
            "/usr/bin/chromium-browser",
            "/usr/bin/chromium",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/google-chrome",
        ]:
            if os.path.exists(chrome_path):
                options.binary_location = chrome_path
                break

        service = None
        for driver_path in [
            "/usr/bin/chromedriver",
            "/usr/lib/chromium/chromedriver",
            "/usr/lib/chromium-browser/chromedriver",
        ]:
            if os.path.exists(driver_path):
                service = Service(executable_path=driver_path)
                break

        if service is None:
            service = Service()

        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(timeout)
        driver.get(url)
        time.sleep(2)
        tmp = output_path + ".tmp"
        driver.save_screenshot(tmp)
        os.replace(tmp, output_path)
        driver.quit()
        return True
    except Exception as e:
        print("[Screenshot] Failed for {}: {}".format(url, e))
        return False
