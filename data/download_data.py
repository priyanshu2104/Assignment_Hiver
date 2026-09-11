"""
download_data.py
----------------
Downloads and filters the Spotify (@SpotifyCares) subset of the
Customer Support on Twitter dataset.

Sources:
  1. HuggingFace Hub – TNE-AI/customer-support-on-twitter-conversation
  2. Direct CSV download / local TWCS CSV fallback
  3. Curated synthetic domain data fallback

Outputs:
  data/spotify_subset.csv   – inbound customer tweets + brand replies, paired
  data/conversations.jsonl  – full multi-turn conversation chains
"""

import os
import json
import re
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(__file__).parent
SUBSET_CSV = DATA_DIR / "spotify_subset.csv"
CONVERSATIONS_JSONL = DATA_DIR / "conversations.jsonl"

SPOTIFY_KEYWORDS = [
    "spotify", "spotifycares", "premium", "playlist", "stream",
    "shuffle", "podcast", "spotify premium", "free trial",
]


def parse_conversation_turns(text: str) -> list[tuple[str, str]]:
    """
    Parses a conversation string into (customer_text, brand_reply) pairs.
    Text format:
        Customer: <text>
        Support: <reply>
        Customer: <text>
        Support: <reply>
    """
    turns = []
    current_role = None
    current_text = []

    for line in text.strip().split("\n"):
        line_clean = line.strip()
        if line_clean.startswith("Customer:"):
            if current_role and current_text:
                turns.append((current_role, " ".join(current_text).strip()))
            current_role = "customer"
            current_text = [line_clean[len("Customer:"):].strip()]
        elif line_clean.startswith("Support:"):
            if current_role and current_text:
                turns.append((current_role, " ".join(current_text).strip()))
            current_role = "support"
            current_text = [line_clean[len("Support:"):].strip()]
        else:
            if current_text is not None and line_clean:
                current_text.append(line_clean)

    if current_role and current_text:
        turns.append((current_role, " ".join(current_text).strip()))

    pairs = []
    for i in range(len(turns) - 1):
        if turns[i][0] == "customer" and turns[i + 1][0] == "support":
            c_text = turns[i][1]
            s_reply = turns[i + 1][1]
            if len(c_text) > 8 and len(s_reply) > 8:
                pairs.append((c_text, s_reply))
    return pairs


def load_from_huggingface() -> pd.DataFrame:
    """Load via HuggingFace datasets library and filter SpotifyCares."""
    try:
        from datasets import load_dataset
        print("Loading dataset from HuggingFace (TNE-AI/customer-support-on-twitter-conversation)...")
        ds = load_dataset(
            "TNE-AI/customer-support-on-twitter-conversation",
            split="train",
        )
        print(f"  Total records in dataset: {len(ds):,}")
        spotify_ds = ds.filter(lambda x: x.get("company") == "SpotifyCares")
        print(f"  SpotifyCares conversations found: {len(spotify_ds):,}")

        pairs = []
        for row in tqdm(spotify_ds, desc="Parsing Spotify conversations"):
            conv_text = row.get("conversation", "")
            conv_id = row.get("conversation_id", "")
            parsed = parse_conversation_turns(conv_text)
            for c_text, b_reply in parsed:
                pairs.append({
                    "conversation_id": conv_id,
                    "customer_text": c_text,
                    "brand_reply": b_reply,
                })

        df = pd.DataFrame(pairs)
        print(f"  Extracted {len(df):,} customer-reply pairs from Spotify conversations.")
        return df
    except Exception as e:
        print(f"  HuggingFace load failed: {e}")
        return None


def load_from_csv_fallback() -> pd.DataFrame:
    """
    Try loading twcs.csv if it exists locally.
    """
    candidates = [
        os.environ.get("TWCS_CSV", ""),
        str(DATA_DIR / "twcs.csv"),
        str(DATA_DIR.parent / "twcs.csv"),
        os.path.expanduser("~/Downloads/twcs.csv"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            print(f"  Loading from local CSV: {path}")
            df = pd.read_csv(path, dtype=str)
            if "inbound" in df.columns:
                outbound = df[df["inbound"].astype(str).str.lower().isin(["false", "0"])]
                mask = outbound["text"].str.lower().str.contains("|".join(SPOTIFY_KEYWORDS), na=False)
                spotify_ids = set(outbound[mask]["author_id"].dropna().unique())
                
                tweet_index = df.set_index("tweet_id")
                brand_tweets = df[df["author_id"].isin(spotify_ids) & ~df["inbound"].astype(str).str.lower().isin(["true", "1"])]
                pairs = []
                for _, brand_row in brand_tweets.iterrows():
                    parent_id = str(brand_row.get("in_response_to_tweet_id", "")).strip()
                    if parent_id in tweet_index.index:
                        customer_row = tweet_index.loc[parent_id]
                        pairs.append({
                            "customer_tweet_id": parent_id,
                            "brand_tweet_id": brand_row["tweet_id"],
                            "customer_text": str(customer_row["text"]),
                            "brand_reply": str(brand_row["text"]),
                        })
                return pd.DataFrame(pairs)
    return None


def build_synthetic_spotify_data() -> pd.DataFrame:
    """
    Fallback: generate a realistic synthetic Spotify dataset.
    """
    print("  Generating synthetic Spotify support data (fallback mode)...")
    synthetic = [
        ("songs keep pausing randomly on my phone", "Hey! Try clearing the cache in your Spotify app settings and let us know if that helps 🎵"),
        ("my music stops every 30 seconds wtf", "That sounds frustrating! Make sure your app is up to date and try a clean reinstall. DM us if the issue continues."),
        ("songs won't play offline even though I downloaded them", "Make sure you're connected to the internet at least once every 30 days to refresh your offline licenses. More help: spoti.fi/offline"),
        ("crossfade stopped working after the update", "Thanks for flagging! Known issue we're working on. Keep an eye on spoti.fi/status for updates."),
        ("spotify skips to next song automatically", "Could be a Sleep Timer or similar setting. Check Settings > Sleep Timer and turn it off if it's on!"),
        ("audio quality sounds terrible on bluetooth", "Try setting audio quality to 'Very High' in Settings > Audio Quality. Also check your BT codec settings."),
        ("shuffle play not working properly, plays same songs", "We're aware some users see this. A workaround: go to Your Library, choose the playlist, and tap shuffle there rather than from Now Playing."),
        ("spotify wont play any songs just loading spinner", "Try logging out and back in, and check your internet connection. If it keeps happening, please DM us your account email!"),
        ("music cuts out every few minutes", "This can happen with unstable WiFi. Try switching to mobile data briefly. Still happening? DM us!"),
        ("can't play songs, says 'can't play the current song'", "Hi! This usually means the track was removed or isn't available in your region. Try searching for it again."),
        ("can't log into my spotify account", "Sorry to hear! Try resetting your password at spoti.fi/reset. If you signed up via Facebook, try the FB login option."),
        ("forgot my spotify password", "No worries! Head to spoti.fi/reset and enter your email to get a reset link."),
        ("spotify says my email doesn't exist", "This can happen if you used a different email. Try all your email addresses or contact us via spoti.fi/contact."),
        ("account hacked, someone changed my email", "We're so sorry! Please contact us immediately via spoti.fi/contact so we can secure your account right away."),
        ("logged out of all devices suddenly", "This can happen after a password change. Log back in and let us know if you need help!"),
        ("two factor auth not working", "2FA codes expire quickly—make sure your device clock is correct. Still issues? DM us!"),
        ("can't remember which email i used for spotify", "Try logging in with your Facebook or Apple account if you used those. Otherwise, DM us and we can help track it down."),
        ("username taken but its mine", "Usernames are unique—sounds like you might have two accounts. DM us and we'll sort it out!"),
        ("charged twice this month for premium", "That's not right! Please DM us your account email and the charge dates so we can investigate and issue a refund if needed."),
        ("cancelled premium but still being charged", "Please DM us your account email and billing details. We'll look into this immediately."),
        ("premium trial ended but I didn't get warned", "Trial end notices are sent by email—check your spam folder. For billing questions, DM us your account info."),
        ("can't cancel my subscription", "You can cancel anytime at spoti.fi/account > Your Plan > Cancel. Still stuck? DM us!"),
        ("student discount not applying", "Make sure you re-verify your student status at spoti.fi/student. Verification is needed each year."),
        ("family plan not working for family members", "All members need to live at the same address. Invite them via spoti.fi/family from the plan manager's account."),
        ("was charged after cancelling", "We're sorry! DM us your email and the charge date and we'll get this resolved for you."),
        ("premium price increased without notice", "Price changes are communicated by email ahead of time. Check your inbox/spam. For specifics, DM us!"),
        ("how do i get a refund", "Refunds are handled case-by-case. DM us your account details and we'll look into it right away!"),
        ("spotify app keeps crashing on my iphone", "Sorry about this! Try force-closing and reopening the app. If that doesn't help, a reinstall usually does the trick."),
        ("app crashes every time i open it", "Let's try a full reinstall—delete the app, restart your phone, and reinstall from the App Store/Play Store."),
        ("spotify black screen on android", "This is a known issue on some Android versions. Try clearing the app cache (Settings > Apps > Spotify > Clear Cache)."),
        ("app freezes on home screen", "Please reinstall the app and let us know your phone model + OS version so we can flag it to our team!"),
        ("can't search anything app freezes", "Clear your cache first (Settings > Apps > Spotify > Storage > Clear Cache). Still happening? DM us!"),
        ("spotify not responding after latest update", "We're aware of some post-update issues. Try reinstalling the latest version from your app store."),
        ("downloaded songs disappeared", "Check that you're logged into the same account and Storage hasn't been cleared. Also verify Offline mode isn't blocking playback."),
        ("can't download songs for offline", "Make sure you have Spotify Premium (downloads are a Premium feature) and enough storage space on your device."),
        ("offline mode not working", "Offline mode requires the app to have synced in the last 30 days. Connect to internet briefly, then try again."),
        ("download limit reached", "Premium allows up to 10,000 songs across 5 devices. Remove some downloads from another device to free up space."),
        ("why was my favorite album removed from spotify", "Labels sometimes pull content from streaming services—this is outside our control. We recommend following the artist for updates."),
        ("song not available in my country", "Content availability varies by region due to licensing. Unfortunately we can't override these restrictions."),
        ("podcast episode disappeared", "Podcasters control their episode availability. Reach out to them directly if an episode went missing."),
        ("playlist I saved is gone", "Check if the playlist owner deleted or made it private. Collaborative playlists you own should still be in Your Library."),
        ("spotify not working on my ps5", "Make sure the PS5 Spotify app is updated. Try uninstalling and reinstalling from the PlayStation Store."),
        ("alexa not playing spotify", "Re-link your Spotify account in the Alexa app (Devices > your Echo > Music). Still issues? DM us!"),
        ("spotify on apple watch not syncing", "Try un-pairing Spotify from Apple Watch and re-pairing. Make sure the Watch app is updated too."),
        ("car bluetooth disconnects from spotify", "This is usually a car/phone Bluetooth issue, not Spotify. Try forgetting and re-pairing the Bluetooth connection."),
        ("smart tv spotify app not loading", "Delete the Spotify app from your TV, restart the TV, and reinstall. Also check for TV firmware updates."),
        ("how do i share a playlist", "Tap the three dots next to your playlist > Share > Copy Link! You can share to any platform from there."),
        ("can i have multiple spotify accounts", "You can, but each needs a different email. Note that Premium is per account."),
        ("spotify wrapped not showing my stats", "Wrapped is available at the end of the year. In the meantime, you can see your stats on spoti.fi/stats!"),
    ]
    return pd.DataFrame(synthetic, columns=["customer_text", "brand_reply"])


def main():
    print("=" * 60)
    print("Spotify Customer Support Data Downloader")
    print("=" * 60)

    pairs_df = None

    # Attempt 1: HuggingFace
    pairs_df = load_from_huggingface()

    # Attempt 2: Local CSV
    if pairs_df is None or len(pairs_df) == 0:
        pairs_df = load_from_csv_fallback()

    # Attempt 3: Synthetic fallback
    if pairs_df is None or len(pairs_df) < 20:
        print("Using synthetic Spotify data (fallback).")
        pairs_df = build_synthetic_spotify_data()

    # Clean text: remove URLs / excessive whitespace
    pairs_df = pairs_df.dropna(subset=["customer_text", "brand_reply"])
    pairs_df["customer_text"] = pairs_df["customer_text"].astype(str).str.strip()
    pairs_df["brand_reply"] = pairs_df["brand_reply"].astype(str).str.strip()
    pairs_df = pairs_df[(pairs_df["customer_text"].str.len() >= 10) & (pairs_df["brand_reply"].str.len() >= 10)]

    # Save
    SUBSET_CSV.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(SUBSET_CSV, index=False)
    print(f"\n[INFO] Saved {len(pairs_df):,} pairs to {SUBSET_CSV}")

    # Also save as JSONL for easy inspection
    with open(CONVERSATIONS_JSONL, "w", encoding="utf-8") as f:
        for _, row in pairs_df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    print(f"[INFO] Saved JSONL to {CONVERSATIONS_JSONL}")


if __name__ == "__main__":
    main()
