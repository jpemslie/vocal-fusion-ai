"""
discord_bot.py — VocalFusion Discord Bot

Drop two audio files in any message or use the slash command /fuse.
The bot fuses them and sends the result back.

Setup:
  1. Create a bot at https://discord.com/developers/applications
  2. Copy the token and set: export DISCORD_TOKEN=your_token_here
  3. Invite URL: OAuth2 → bot + applications.commands scopes
     Permissions: Send Messages, Attach Files, Read Message History, Add Reactions
  4. python discord_bot.py

Usage in Discord:
  /fuse  (attach beat + vocal as two files in the same command)
  — or —
  Drop two audio files in a message → bot auto-detects and fuses them
"""
from __future__ import annotations

import asyncio
import io
import os
import sys
import tempfile
from pathlib import Path

import aiofiles
import aiohttp
import discord
from discord import app_commands
from discord.ext import commands

API_BASE = os.environ.get("VF_API_BASE", "http://localhost:8000")
TOKEN    = os.environ.get("DISCORD_TOKEN", "")

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aiff"}

HELP_TEXT = """
**VocalFusion Bot** 🎛️
Fuse a vocal and a beat into a professional radio mix.

**How to use:**
Attach **two audio files** to any message (beat + vocal, in any order).
The bot auto-detects which is which and sends back the mix.

Or use the slash command:
`/fuse` — attach both files in the command

**Options (slash command only):**
`/fuse pro:False` — skip optimization (faster, lower quality)
`/fuse seed:123`  — set a specific random seed

**Supported formats:** WAV · MP3 · FLAC · OGG · M4A · AAC · AIFF
""".strip()


# ---------------------------------------------------------------------------
# Bot setup
# ---------------------------------------------------------------------------

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)


# ---------------------------------------------------------------------------
# Core fuse logic
# ---------------------------------------------------------------------------

async def _download_attachment(att: discord.Attachment, dest: Path) -> None:
    """Download a Discord attachment to a local path."""
    async with aiohttp.ClientSession() as session:
        async with session.get(att.url) as resp:
            resp.raise_for_status()
            async with aiofiles.open(dest, "wb") as f:
                await f.write(await resp.read())


async def _run_fuse(
    interaction_or_msg,
    attachments: list[discord.Attachment],
    pro_mode: bool = True,
    seed: int = 42,
) -> None:
    """Download files, POST to /fuse, poll status, reply with result."""

    # ── validate ──────────────────────────────────────────────────────────
    audio_atts = [a for a in attachments
                  if Path(a.filename).suffix.lower() in AUDIO_EXTS]
    if len(audio_atts) < 2:
        msg = "❌ Please attach **two audio files** (beat + vocal)."
        if hasattr(interaction_or_msg, "followup"):
            await interaction_or_msg.followup.send(msg, ephemeral=True)
        else:
            await interaction_or_msg.reply(msg)
        return

    att_a, att_b = audio_atts[0], audio_atts[1]

    # ── send initial status ───────────────────────────────────────────────
    embed = discord.Embed(
        title="🎛️ VocalFusion",
        description=f"**{att_a.filename}** × **{att_b.filename}**",
        color=discord.Color.blurple(),
    )
    embed.add_field(name="Status", value="⏳ Uploading files…", inline=False)
    embed.add_field(name="Mode",   value="Pro" if pro_mode else "Fast", inline=True)
    embed.add_field(name="Seed",   value=str(seed), inline=True)

    if hasattr(interaction_or_msg, "followup"):
        status_msg = await interaction_or_msg.followup.send(embed=embed)
    else:
        status_msg = await interaction_or_msg.reply(embed=embed)

    async def _update(text: str, color: discord.Color | None = None) -> None:
        embed.set_field_at(0, name="Status", value=text, inline=False)
        if color:
            embed.color = color
        try:
            await status_msg.edit(embed=embed)
        except Exception:
            pass

    # ── download to temp dir ──────────────────────────────────────────────
    with tempfile.TemporaryDirectory(prefix="vf_discord_") as tmp:
        path_a = Path(tmp) / att_a.filename
        path_b = Path(tmp) / att_b.filename

        try:
            await asyncio.gather(
                _download_attachment(att_a, path_a),
                _download_attachment(att_b, path_b),
            )
        except Exception as e:
            await _update(f"❌ Download failed: {e}", discord.Color.red())
            return

        # ── POST /fuse ─────────────────────────────────────────────────────
        await _update("⚙️ Starting fuse job…")
        job_id = None
        try:
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()
                data.add_field("song_a", open(path_a, "rb"),
                               filename=att_a.filename, content_type="audio/wav")
                data.add_field("song_b", open(path_b, "rb"),
                               filename=att_b.filename, content_type="audio/wav")
                data.add_field("seed",         str(seed))
                data.add_field("direct_vocal", "false")
                data.add_field("pro_mode",     "true" if pro_mode else "false")

                async with session.post(f"{API_BASE}/fuse", data=data) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        await _update(f"❌ API error {resp.status}: {body[:200]}",
                                      discord.Color.red())
                        return
                    job_id = (await resp.json())["job_id"]
        except aiohttp.ClientConnectorError:
            await _update("❌ Cannot reach VocalFusion API. Is it running?",
                          discord.Color.red())
            return
        except Exception as e:
            await _update(f"❌ Fuse request failed: {e}", discord.Color.red())
            return

        # ── poll /status ───────────────────────────────────────────────────
        prev_message = ""
        attempts_shown: set[int] = set()

        while True:
            await asyncio.sleep(3)
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{API_BASE}/status/{job_id}") as resp:
                        state = await resp.json()
            except Exception:
                continue

            status   = state.get("status", "queued")
            progress = state.get("progress", 0)
            message  = state.get("message", "")
            genre    = state.get("genre", "")
            attempts = state.get("attempts", [])

            # Build attempt log lines for new attempts
            new_lines = []
            for att_info in attempts:
                n = att_info.get("attempt", att_info.get("n", 0))
                if n not in attempts_shown:
                    attempts_shown.add(n)
                    cs = att_info.get("chart_score", "?")
                    cg = att_info.get("chart_grade", "?")
                    new_lines.append(f"`#{n}` → **{cs}/100** ({cg})")

            genre_tag = f" · {genre.upper()}" if genre else ""
            bar = _progress_bar(progress)
            status_text = f"{bar} {progress}%{genre_tag}\n_{message}_"
            if new_lines:
                status_text += "\n" + "\n".join(new_lines)

            if status_text != prev_message:
                await _update(status_text)
                prev_message = status_text

            if status == "done":
                break
            if status == "error":
                err = state.get("message", "Unknown error")
                await _update(f"❌ {err}", discord.Color.red())
                return

        # ── download result ────────────────────────────────────────────────
        await _update("📥 Downloading mix…")
        result_path = Path(tmp) / "mix.wav"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{API_BASE}/download/{job_id}") as resp:
                    resp.raise_for_status()
                    async with aiofiles.open(result_path, "wb") as f:
                        await f.write(await resp.read())
        except Exception as e:
            await _update(f"❌ Download failed: {e}", discord.Color.red())
            return

        # ── send final result ──────────────────────────────────────────────
        state_final = state  # last polled state
        cs    = state_final.get("chart_score", "?")
        cg    = state_final.get("chart_grade", "?")
        genre = state_final.get("genre", "")
        n_att = len(state_final.get("attempts", []))

        embed.color = discord.Color.green()
        embed.set_field_at(
            0, name="Status",
            value=f"✅ Done · **{cs}/100** ({cg})\n{n_att} attempt(s) · {genre.upper() if genre else ''}",
            inline=False,
        )
        await status_msg.edit(embed=embed)

        file_size_mb = result_path.stat().st_size / 1_048_576
        safe_name = f"vf_{Path(att_a.filename).stem}_x_{Path(att_b.filename).stem}.wav"

        if file_size_mb > 25:
            # Discord 25 MB limit — send download link instead
            await status_msg.reply(
                f"🎵 Mix ready! File is {file_size_mb:.1f} MB — too large for Discord.\n"
                f"Download: `{API_BASE}/download/{job_id}`"
            )
        else:
            await status_msg.reply(
                f"🎵 **{cs}/100** ({cg})",
                file=discord.File(str(result_path), filename=safe_name),
            )


def _progress_bar(pct: int, width: int = 10) -> str:
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Slash command: /fuse
# ---------------------------------------------------------------------------

@bot.tree.command(name="fuse", description="Fuse a beat and vocal into a radio mix")
@app_commands.describe(
    beat="The instrumental / beat file",
    vocal="The vocal file",
    pro="Use Bayesian optimizer for best quality (default: True)",
    seed="Random seed (default: 42)",
)
async def fuse_command(
    interaction: discord.Interaction,
    beat: discord.Attachment,
    vocal: discord.Attachment,
    pro: bool = True,
    seed: int = 42,
) -> None:
    await interaction.response.defer(thinking=True)
    await _run_fuse(interaction, [beat, vocal], pro_mode=pro, seed=seed)


# ---------------------------------------------------------------------------
# Message listener: drop two files in any message
# ---------------------------------------------------------------------------

@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return

    audio_atts = [a for a in message.attachments
                  if Path(a.filename).suffix.lower() in AUDIO_EXTS]

    if len(audio_atts) >= 2:
        await _run_fuse(message, audio_atts)
        return

    await bot.process_commands(message)


# ---------------------------------------------------------------------------
# Text commands
# ---------------------------------------------------------------------------

@bot.command(name="help", aliases=["vf", "vocalfusion"])
async def help_command(ctx: commands.Context) -> None:
    await ctx.reply(HELP_TEXT)


@bot.command(name="status")
async def status_command(ctx: commands.Context, job_id: str) -> None:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{API_BASE}/status/{job_id}") as resp:
                state = await resp.json()
        s = state.get("status")
        p = state.get("progress", 0)
        cs = state.get("chart_score", "?")
        await ctx.reply(f"`{job_id[:8]}…` — **{s}** {p}% score={cs}")
    except Exception as e:
        await ctx.reply(f"❌ {e}")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@bot.event
async def on_ready() -> None:
    await bot.tree.sync()
    print(f"VocalFusion bot online as {bot.user} ({bot.user.id})")
    print(f"API: {API_BASE}")
    print("Slash commands synced. Invite URL scopes needed: bot + applications.commands")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not TOKEN:
        print("ERROR: Set DISCORD_TOKEN environment variable.")
        print("  export DISCORD_TOKEN=your_bot_token_here")
        sys.exit(1)
    bot.run(TOKEN)
