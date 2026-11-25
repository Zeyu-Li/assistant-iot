"""Command line: ``assistant chat`` (text REPL) and friends."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from assistant.adapters.control.unix_socket import send
from assistant.adapters.google.oauth import login
from assistant.app import DECISION_ADAPTERS, build, build_assistant, build_tool_clients, build_tts, build_voice_loop
from assistant.config import Profile, load_profile, resolve_profile_path
from assistant.core.listening import ListeningMode
from assistant.core.registry import Registry
from assistant.eval_router import load_examples, report, score

REPL_HELP = """Type what you would say. Commands:
  /overheard <text>   speech without the wake word (tests open-mic behaviour)
  /mode <muted|wake_word|open_mic>
  /quit"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="assistant")
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--profile", default="mac-dev", help="profile name in config/profiles, or a YAML path")
    parser.add_argument("-v", "--verbose", action="store_true", help="log routing decisions")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("chat", help="talk to the assistant by typing")
    evaluate = commands.add_parser("eval-router", help="measure routing accuracy on a labelled set")
    evaluate.add_argument("--examples", type=Path, default=Path("tests/routing_eval/utterances.yaml"))
    evaluate.add_argument(
        "--all-capabilities", action="store_true", help="ignore the profile's disabled_capabilities"
    )
    commands.add_parser("listen", help="run the voice assistant on the microphone and speaker")
    mode = commands.add_parser("mode", help="show or change the running assistant's listening mode")
    mode.add_argument("name", nargs="?", choices=[m.value for m in ListeningMode])
    say = commands.add_parser("say", help="speak a sentence with the profile's voice (speaker check)")
    say.add_argument("text")
    commands.add_parser("google-login", help="authorise Gmail and Calendar access (opens a browser once)")
    tools = commands.add_parser("mcp-tools", help="list the tools an MCP server offers")
    tools.add_argument("server", help="server name from the profile's mcp_servers")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    profile = load_profile(resolve_profile_path(args.profile, args.config_dir))
    capabilities = args.config_dir / "capabilities.yaml"
    if args.command == "chat":
        return asyncio.run(_chat(profile, capabilities))
    if args.command == "eval-router":
        return asyncio.run(_eval_router(profile, capabilities, args.examples, args.all_capabilities))
    if args.command == "listen":
        return asyncio.run(_listen(profile, capabilities))
    if args.command == "mode":
        if profile.voice is None:
            print("this profile has no 'voice' section")
            return 1
        line = f"mode {args.name}" if args.name else "mode"
        print(asyncio.run(send(profile.voice.control_socket, line)))
        return 0
    if args.command == "say":
        if profile.voice is None:
            print("this profile has no 'voice' section")
            return 1
        from assistant.adapters.audio.sounddevice_io import SoundDeviceSpeaker

        audio = build_tts(profile.voice.tts).synthesize(args.text)
        asyncio.run(SoundDeviceSpeaker(profile.voice.output_device).play(audio))
        return 0
    if args.command == "google-login":
        stored = login(profile.google_oauth)
        print(f"Authorised. Scopes: {', '.join(stored.scopes)}. Stored in {profile.google_oauth.token_store}.")
        return 0
    if args.command == "mcp-tools":
        return asyncio.run(_mcp_tools(profile, args.server))
    return 2


async def _listen(profile: Profile, capabilities: Path) -> int:
    assistant = await build_assistant(profile, capabilities)
    voice_loop, control = build_voice_loop(profile, assistant)
    server = await control.serve()
    phrases = " / ".join(profile.voice.wake_phrases) if profile.voice else ""
    print(f"Listening ({assistant.listening.mode.value}). Wake phrase: {phrases}. Ctrl-C to stop.")
    try:
        await voice_loop.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        server.close()
    return 0


async def _mcp_tools(profile: Profile, server: str) -> int:
    clients = build_tool_clients(profile)
    if server not in clients:
        print(f"no MCP server '{server}' in this profile; defined: {', '.join(clients) or 'none'}")
        return 1
    for spec in sorted(await clients[server].list_tools(), key=lambda s: s.name):
        params = ", ".join(spec.input_schema.get("properties", {}))
        print(f"{spec.name}({params})\n    {spec.description.splitlines()[0] if spec.description else ''}")
    return 0


async def _eval_router(profile: Profile, capabilities: Path, examples_path: Path, all_capabilities: bool) -> int:
    registry = Registry.load(capabilities, disabled=() if all_capabilities else profile.disabled_capabilities)
    decider = build(DECISION_ADAPTERS, profile.decision, profile, "decision")
    examples = load_examples(examples_path, registry)
    results = await score(decider, registry, examples, profile.router.include_examples)
    print(report(results, profile.router.route_threshold))
    return 0


async def _chat(profile: Profile, capabilities: Path) -> int:
    assistant = await build_assistant(profile, capabilities)
    print(REPL_HELP)
    loop = asyncio.get_running_loop()
    while True:
        try:
            line = (await loop.run_in_executor(None, input, f"[{assistant.listening.mode.value}] you> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line == "/quit":
            return 0
        if line.startswith("/mode "):
            try:
                assistant.listening.set_mode(ListeningMode(line.split(maxsplit=1)[1]))
            except ValueError:
                print("modes: " + ", ".join(m.value for m in ListeningMode))
            continue
        overheard = line.startswith("/overheard ")
        text = line.split(maxsplit=1)[1] if overheard else line
        reply = await assistant.handle_utterance(text, wake_word_detected=not overheard)
        print(f"assistant> {reply.text}" if reply else "assistant> (silent)")
