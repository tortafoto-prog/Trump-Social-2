#!/usr/bin/env python3
"""
Trump Social Media Scraper (Roll Call Factbase) with Hungarian LLM Translation
Scrapes Donald Trump's posts from Roll Call Factbase, translates to Hungarian, and posts to Discord.
"""

import os
import time
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

from anthropic import Anthropic
from discord_webhook import DiscordWebhook, DiscordEmbed
from playwright.sync_api import sync_playwright


# Configuration from environment variables
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "300"))
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

        with sync_playwright() as p:
            print("⏳ Opening headless browser to scrape Roll Call...")
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = context.new_page()

            try:
                page.goto(ROLLCALL_URL, wait_until="networkidle", timeout=60000)
                page.wait_for_selector("div.rounded-xl.border", timeout=30000)
                time.sleep(2)

                extracted_data = page.evaluate("""() => {
                    const posts = [];
                    const cards = document.querySelectorAll('div.rounded-xl.border');

                    cards.forEach(card => {
                        const contentEl = card.querySelector('div.text-sm.font-medium.whitespace-pre-wrap');
                        const content = contentEl ? contentEl.innerText.trim() : "";

                        const truthLinkEl = Array.from(card.querySelectorAll('a')).find(a => a.innerText.includes('View on Truth Social'));
                        const url = truthLinkEl ? truthLinkEl.href : "";

                        const timeEl = Array.from(card.querySelectorAll('div')).find(div => div.innerText.includes('@') && div.innerText.includes('ET'));
                        const timestamp_str = timeEl ? timeEl.innerText.trim() : "";

                        let id = "";
                        if (url) {
                            const matches = url.match(/posts\\/(\\d+)/);
                            id = matches ? matches[1] : url;
                        } else {
                            id = btoa(content.substring(0, 50) + timestamp_str).replace(/[^a-zA-Z0-9]/g, "");
                        }

                        if (content || url) {
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
                print(f"✓ Found {len(posts)} posts on Roll Call")

            except Exception as e:
                print(f"✗ Error during scraping: {e}")
            finally:
                browser.close()

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

    def translate_to_hungarian(self, text: str) -> str:
        """Translate text to Hungarian while preserving URLs, hashtags, and mentions"""
        text = self.clean_text(text)

        if not text or not text.strip():
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
                print("⚠ Warning: URL mismatch in translation.")

            print(f"✓ Translated text ({len(text)} -> {len(translated)} chars)")
            return translated

        except Exception as e:
            print(f"✗ Translation error: {e}")
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
            embed.set_author(
                name="Donald J. Trump",
                icon_url="https://truthsocial.com/avatars/original/missing.png"
            )
            embed.set_title("🇺🇸 Új Truth Social bejegyzés - Donald Trump")

            description_parts = []
            if original_text:
                description_parts.append("**Eredeti szöveg:**")
                description_parts.append(original_text)
                description_parts.append("")

            if translated_text:
                description_parts.append("**Magyar fordítás:**")
                description_parts.append(translated_text)

            if description_parts:
                embed.set_description("\n".join(description_parts))

            post_url = post_data.get("url", "")
            if post_url:
                embed.add_embed_field(
                    name="🔗 Eredeti bejegyzés",
                    value=f"[Link a Truth Social-hoz]({post_url})",
                    inline=False
                )

            timestamp_str = post_data.get("timestamp_str", "")
            if timestamp_str:
                embed.set_footer(text=f"🤖 Generated by TotAI AI • {timestamp_str}")
            else:
                embed.set_footer(text="🤖 Generated by TotAI AI")

            embed.set_color(color=0x1DA1F2)

            webhook.add_embed(embed)
            response = webhook.execute()

            if response.status_code in [200, 204]:
                print("✓ Posted to Discord successfully")
            else:
                print(f"✗ Discord post failed with status {response.status_code}")

        except Exception as e:
            print(f"✗ Error posting to Discord: {e}")


class StateManager:
    """Manages persistent state for tracking processed posts"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "last_id.txt"

    def load_last_id(self) -> Optional[str]:
        """Load the last processed post ID"""
        try:
            if self.state_file.exists():
                last_id = self.state_file.read_text().strip()
                print(f"✓ Loaded last processed ID: {last_id}")
                return last_id
            else:
                print("✓ No previous state found, starting fresh")
                return None
        except Exception as e:
            print(f"✗ Error loading state: {e}")
            return None

    def save_last_id(self, post_id: str):
        """Save the last processed post ID"""
        try:
            self.state_file.write_text(post_id)
            print(f"✓ Saved last processed ID: {post_id}")
        except Exception as e:
            print(f"✗ Error saving state: {e}")


def validate_environment():
    """Validate required environment variables"""
    missing = []

    if not DISCORD_WEBHOOK_URL:
        missing.append("DISCORD_WEBHOOK_URL")
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")

    if missing:
        print(f"✗ Missing required environment variables: {', '.join(missing)}")
        return False

    print("✓ Environment variables validated")
    return True


def main():
    """Main execution loop"""
    print("=" * 60)
    print("Trump Scraper (Roll Call Aggregator Mode)")
    print("=" * 60)

    if not validate_environment():
        return

    scraper = RollCallScraper(headless=True)
    translator = Translator(ANTHROPIC_API_KEY, ANTHROPIC_MODEL)
    discord_poster = DiscordPoster(DISCORD_WEBHOOK_URL)
    state_manager = StateManager(DATA_DIR)

    last_id = state_manager.load_last_id()
    if FORCE_REPROCESS:
        print("⚠ FORCE_REPROCESS enabled: Ignoring saved state for this run!")
        last_id = None

    print(f"\n✓ Starting monitoring loop (interval: {CHECK_INTERVAL}s)")
    print("-" * 60)

    try:
        while True:
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Checking for new posts on Roll Call...")

            if FORCE_REPROCESS:
                print("⚠ FORCE_REPROCESS is enabled - ignoring last_id for this check")
                check_last_id = None
            else:
                check_last_id = last_id

            posts = scraper.scrape_latest_posts()

            # Roll Call returns posts in DESC order (newest first)
            # We want to process them in ASC order (oldest first)
            posts.reverse()

            # Find new posts (posts after the last processed ID)
            new_posts = []
            found_last_id = False if check_last_id else True

            for post in posts:
                if not found_last_id:
                    if post['id'] == check_last_id:
                        found_last_id = True
                    continue
                new_posts.append(post)

            if new_posts:
                print(f"Found {len(new_posts)} new posts to process.")
                for post in new_posts:
                    print(f"Processing post {post['id']}...")

                    original_text = translator.clean_text(post.get('content', ""))
                    translated = ""
                    if original_text:
                        translated = translator.translate_to_hungarian(original_text)

                    discord_poster.post_to_discord(post, translated, original_text)

                    if not FORCE_REPROCESS:
                        last_id = post['id']
                        state_manager.save_last_id(last_id)
                    time.sleep(2)
            else:
                print("✓ No new posts found (since last check)")

            print(f"\n⏳ Waiting {CHECK_INTERVAL} seconds until next check...")
            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n✓ Shutting down gracefully...")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
