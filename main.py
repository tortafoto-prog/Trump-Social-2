#!/usr/bin/env python3
"""
Trump Social Media Scraper (Roll Call Factbase) with Hungarian LLM Translation
Scrapes Donald Trump's posts from Roll Call Factbase, translates to Hungarian, and posts to Discord.
"""

import os
import sys
import time
import re
from pathlib import Path
from typing import List, Dict, Any

from anthropic import Anthropic
from discord_webhook import DiscordWebhook, DiscordEmbed
from playwright.sync_api import sync_playwright


def log(message: str):
    """Print with flush for immediate output in Docker"""
    print(message, flush=True)


# Configuration from environment variables
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "60"))
DATA_DIR = os.getenv("DATA_DIR", "/data")
FORCE_REPROCESS = os.getenv("FORCE_REPROCESS", "false").lower() == "true"

# Target configuration
ROLLCALL_URL = "https://rollcall.com/factbase/trump/topic/social/?platform=all&sort=date&sort_order=desc&page=1"

# Translation system prompt
TRANSLATION_SYSTEM_PROMPT = """Te egy professzionális fordító vagy, aki gyönyörű, természetes magyarsággal dolgozik.

Feladatod: Fordítsd le ezt a közösségi média bejegyzést angolról magyarra!

FORDÍTÁSI ELVEK:
- Használj természetes, gördülékeny magyar nyelvezetet - ne szó szerinti fordítást!
- Tartsd meg az eredeti hangnemet és stílust (ha harcos, az maradjon harcos; ha informális, az informális)
- Politikai/publicisztikai szövegekhez használj kifejező, erőteljes magyar nyelvezetet
- NE fordítsd le: URL-eket, hashtag-eket (#), említéseket (@)
- Ha van idiomatikus angol kifejezés, használj neki megfelelő magyar megfelelőt
- Kerüld a magyartalanságokat és az angolból átvett mondatszerkezeteket

VÁLASZ: Csak a lefordított szöveget add vissza, semmi mást!"""


class RollCallScraper:
    """Handles scraping of Trump's posts from the Roll Call Factbase aggregator"""

    def __init__(self, headless: bool = True):
        self.headless = headless

    def scrape_latest_posts(self) -> List[Dict[str, Any]]:
        """Scrape the latest posts from Roll Call using Playwright"""
        posts = []

        try:
            with sync_playwright() as p:
                log("⏳ Opening headless browser to scrape Roll Call...")
                browser = p.chromium.launch(
                    headless=self.headless,
                    args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
                )
                log("✓ Browser launched successfully")

                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
                page = context.new_page()
                log("✓ Page created, navigating to Roll Call...")

                try:
                    # Add cache buster to URL
                    cache_buster = int(time.time())
                    final_url = f"{ROLLCALL_URL}&t={cache_buster}"
                    
                    # Use domcontentloaded instead of networkidle (faster, more reliable)
                    page.goto(final_url, wait_until="domcontentloaded", timeout=90000)
                    log("✓ DOM loaded, waiting for posts to render...")

                    # Wait for the actual post content to appear
                    # Verify selector: div.rounded-xl.border
                    page.wait_for_selector("div.rounded-xl.border", timeout=60000)
                    log("✓ Post cards found, waiting for content to fully load...")

                    # Wait a bit more for Alpine.js to render content
                    time.sleep(5)

                    extracted_data = page.evaluate("""() => {
                        const posts = [];
                        const cards = document.querySelectorAll('div.rounded-xl.border');

                        cards.forEach(card => {
                            // Only process cards that have a Truth Social link
                            const truthLinkEl = Array.from(card.querySelectorAll('a')).find(a =>
                                a.innerText.includes('View on Truth Social') && a.href.includes('truthsocial.com')
                            );

                            if (!truthLinkEl) return; // Skip non-post cards

                            const url = truthLinkEl.href;
                            const contentEl = card.querySelector('div.text-sm.font-medium.whitespace-pre-wrap');
                            const content = contentEl ? contentEl.innerText.trim() : "";

                            const timeEl = Array.from(card.querySelectorAll('div')).find(div =>
                                div.innerText.includes('@') && div.innerText.includes('ET')
                            );
                            const timestamp_str = timeEl ? timeEl.innerText.trim() : "";

                            // Extract ID from URL
                            const matches = url.match(/posts\\/(\\d+)/);
                            const id = matches ? matches[1] : "";

                            if (id && (content || url)) {
                                posts.push({
                                    id: id,
                                    url: url,
                                    content: content,
                                    timestamp_str: timestamp_str,
                                    created_at: new Date().toISOString()
                                });
                            }
                        });
                        return posts;
                    }""")

                    posts = extracted_data
                    log(f"✓ Found {len(posts)} posts on Roll Call")

                except Exception as e:
                    log(f"✗ Error during scraping: {e}")
                finally:
                    browser.close()
                    log("✓ Browser closed")

        except Exception as e:
            log(f"✗ Playwright error: {e}")

        return posts


class Translator:
    """Handles translation using Anthropic Claude API"""

    def __init__(self, api_key: str, model: str):
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def clean_text(self, text: str) -> str:
        """Basic text cleanup"""
        if not text:
            return ""
        return text.strip()

    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text to preserve them"""
        url_pattern = r'https?://[^\s]+'
        return re.findall(url_pattern, text)

    def has_translatable_content(self, text: str) -> bool:
        """Check if text has content worth translating (not just URLs/links)"""
        if not text:
            return False
        # Remove URLs from text
        text_without_urls = re.sub(r'https?://[^\s]+', '', text).strip()
        # Check if there's meaningful text left (at least 10 chars)
        return len(text_without_urls) >= 10

    def translate_to_hungarian(self, text: str) -> str:
        """Translate text to Hungarian while preserving URLs, hashtags, and mentions"""
        text = self.clean_text(text)

        if not text or not text.strip():
            return ""

        # Skip translation if text is just URLs/links
        if not self.has_translatable_content(text):
            log("⏭ Skipping translation: text is only URLs/links")
            return ""

        try:
            original_urls = self.extract_urls(text)

            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=TRANSLATION_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": text}
                ],
                temperature=0.3
            )

            translated = response.content[0].text.strip()

            translated_urls = self.extract_urls(translated)
            if set(original_urls) != set(translated_urls):
                log("⚠ Warning: URL mismatch in translation.")

            log(f"✓ Translated text ({len(text)} -> {len(translated)} chars)")
            return translated

        except Exception as e:
            log(f"✗ Translation error: {e}")
            return text


class DiscordPoster:
    """Handles posting to Discord via webhook"""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def post_to_discord(self, post_data: Dict[str, Any], translated_text: str, original_text: str = ""):
        """Post translated content to Discord with both original and translated text"""
        try:
            webhook = DiscordWebhook(url=self.webhook_url)

            embed = DiscordEmbed()
            embed.set_title("🇺🇸 Új Truth Social bejegyzés - Donald Trump")

            description_parts = []
            
            # Truncate if too long (Discord limit is ~4096 for description)
            if original_text and len(original_text) > 1800:
                original_text = original_text[:1800] + "... [tovább az eredeti linken]"
            
            if original_text:
                description_parts.append("**Eredeti szöveg:**")
                description_parts.append(original_text)
                description_parts.append("")

            if translated_text and len(translated_text) > 2000:
                translated_text = translated_text[:2000] + "... [tovább az eredeti linken]"

            if translated_text:
                description_parts.append("**Magyar fordítás:**")
                description_parts.append(translated_text)

            full_description = "\n".join(description_parts)
            if len(full_description) > 4096:
                 full_description = full_description[:4093] + "..."

            if description_parts:
                embed.set_description(full_description)

            # Add extra space before the link
            post_url = post_data.get("url", "")
            if post_url:
                 embed.add_embed_field(
                    name="🔗 Eredeti bejegyzés",
                    value=f"[Link a Truth Social-hoz]({post_url})",
                    inline=False
                )

            # Footer with original timestamp
            timestamp_str = post_data.get("timestamp_str", "")
            if timestamp_str:
                embed.set_footer(text=f"🤖 Generated by TotAI AI • {timestamp_str}")
            else:
                # Fallback to current time
                from datetime import datetime
                import pytz
                budapest_tz = pytz.timezone('Europe/Budapest')
                budapest_time = datetime.now(budapest_tz).strftime("%Y.%m.%d. %H:%M")
                embed.set_footer(text=f"🤖 Generated by TotAI AI • {budapest_time}")

            embed.set_color(color=0x1DA1F2)

            webhook.add_embed(embed)
            response = webhook.execute()

            if response.status_code in [200, 204]:
                log("✓ Posted to Discord successfully")
            else:
                log(f"✗ Discord post failed with status {response.status_code}")

        except Exception as e:
            log(f"✗ Error posting to Discord: {e}")


class StateManager:
    """Manages persistent state for tracking processed posts"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "last_id.txt"

    def load_last_id(self) -> str:
        """Load the last processed post ID"""
        try:
            if self.state_file.exists():
                last_id = self.state_file.read_text().strip()
                log(f"✓ Loaded last processed ID: {last_id}")
                return last_id
        except Exception as e:
            log(f"⚠ Could not load state: {e}")
        return None

    def save_last_id(self, last_id: str):
        """Save the last processed post ID"""
        try:
            self.state_file.write_text(str(last_id))
            # log(f"✓ Saved last processed ID: {last_id}") # Too verbose for every post
        except Exception as e:
            log(f"⚠ Could not save state: {e}")


def validate_environment():
    """Validate required environment variables"""
    missing = []

    if not DISCORD_WEBHOOK_URL:
        missing.append("DISCORD_WEBHOOK_URL")
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")

    if missing:
        log(f"✗ Missing required environment variables: {', '.join(missing)}")
        return False

    log("✓ Environment variables validated")
    return True


def main():
    """Main execution loop"""
    log("=" * 60)
    log("Trump Scraper (Roll Call Aggregator Mode) - v2")
    log("=" * 60)

    if not validate_environment():
        return

    # Check if data directory exists and is writable
    data_path = Path(DATA_DIR)
    log(f"Data directory: {DATA_DIR}")
    try:
        data_path.mkdir(parents=True, exist_ok=True)
        test_file = data_path / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
        log(f"✓ Data directory is writable")
    except Exception as e:
        log(f"⚠ WARNING: Data directory not writable: {e}")
        log(f"⚠ State persistence will NOT work - duplicates may occur!")

    scraper = RollCallScraper(headless=True)
    translator = Translator(ANTHROPIC_API_KEY, ANTHROPIC_MODEL)
    discord_poster = DiscordPoster(DISCORD_WEBHOOK_URL)
    state_manager = StateManager(DATA_DIR)

    last_id = state_manager.load_last_id()
    
    if FORCE_REPROCESS:
        log("⚠ FORCE_REPROCESS enabled: Ignoring saved state for this run!")
        check_last_id = None
    else:
        check_last_id = last_id

    log(f"\n✓ Starting monitoring loop (interval: {CHECK_INTERVAL}s)")
    log("-" * 60)

    try:
        while True:
            log(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Checking for new posts on Roll Call...")

            posts = scraper.scrape_latest_posts()

            # Roll Call returns posts in DESC order (newest first)
            # We want to process them in ASC order (oldest first)
            posts.reverse()

            # Find new posts (posts with ID > last_id)
            new_posts = []
            
            # Helper to convert to int safely
            def to_int(val):
                try:
                    return int(val)
                except:
                    return None

            last_id_int = to_int(check_last_id) if check_last_id else None

            if not check_last_id:
                # First run / No state: Process ONLY the newest post to initialize state
                # posts list is ASC (oldest -> newest), so posts[-1] is the newest.
                if posts:
                    newest_post = posts[-1]
                    new_posts = [newest_post]
                    log(f"First run (or no state): Processing only the newest post ({newest_post['id']}) to initialize.")
            else:
                # Normal operation: Filter posts newer than last_id
                for post in posts:
                    post_id_int = to_int(post['id'])
                    
                    if last_id_int and post_id_int:
                        # Robust Numeric Comparison
                        if post_id_int > last_id_int:
                            new_posts.append(post)
                    elif check_last_id:
                         # Fallback for non-numeric IDs
                         if str(post['id']) > str(check_last_id):
                              new_posts.append(post)

            if new_posts:
                log(f"Found {len(new_posts)} new posts to process.")
                for post in new_posts:
                    log(f"Processing post {post['id']}...")

                    original_text = translator.clean_text(post.get('content', ""))
                    translated = ""
                    if original_text:
                        translated = translator.translate_to_hungarian(original_text)

                    discord_poster.post_to_discord(post, translated, original_text)
                    
                    # Update state immediately
                    check_last_id = post['id']
                    state_manager.save_last_id(check_last_id)
                    
                    time.sleep(2)
            else:
                log("✓ No new posts found (since last check)")

            log(f"\n⏳ Waiting {CHECK_INTERVAL} seconds until next check...")
            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        log("\n\n✓ Shutting down gracefully...")
    except Exception as e:
        log(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
