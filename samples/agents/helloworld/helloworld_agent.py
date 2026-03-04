#!/usr/bin/env python3
"""
HelloWorld Agent using Ephapsys SDK.

Workflow:
1. Load agent from environment (agent_id, API base/key).
2. Verify agent (status, certs, models) and personalize agent if required.
3. Prepare runtime (download/cache artifacts + decrypt ECM).
4. Enter a loop:
   - Re-verify agent status each cycle.
   - Prompt the user for text input.
   - Send input to the agent and print its response.
   - Exit gracefully on 'exit' or Ctrl+C.
"""

import os, sys, time, warnings, logging
from ephapsys.agent import TrustedAgent

# Suppress HF generation warnings (e.g., pad_token_id notice) for cleaner demo output
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`.*")

def main():
    # Optional GGUF mode (kept commented for default cleanliness):
    # If your language artifact is .gguf and you want llama.cpp CLI fallback,
    # uncomment and set these before creating TrustedAgent.
    #
    # if os.getenv("MODEL_RUNTIME", "").lower() == "gguf":
    #     os.environ.setdefault("AOC_LLAMA_CPP_CLI", "llama-cli")
    #     os.environ.setdefault("AOC_GGUF_CTX", "2048")
    #     os.environ.setdefault("AOC_GGUF_MAX_NEW_TOKENS", "256")

    agent = TrustedAgent.from_env()
    # Optional A2A mode (kept commented for default cleanliness):
    # from ephapsys import A2AClient
    # a2a = A2AClient.from_env()
    # peer_agent = os.getenv("A2A_PEER_AGENT_ID", "").strip()

    print("=== Step 1: Verify Agent ===")
    try:
        ok, report = agent.verify()
    except RuntimeError as e:
        if "404" in str(e):
            print(f"[HelloWorld] ❌ Agent template '{agent.agent_id}' not found in backend.")
            print("[HelloWorld] Please create it in the AOC before running this sample.")
            sys.exit(1)
        else:
            raise

    # Personalize if needed
    if not ok:
        status = agent.get_status()
        is_personalized = status.get("state", {}).get("personalized", False) or status.get("personalized", False)
        if not is_personalized:
            anchor = os.getenv("PERSONALIZE_ANCHOR", "tpm")
            print(f"[HelloWorld] Agent not personalized; running personalize(anchor={anchor})...")
            agent.personalize(anchor=anchor)
            print(f"DEBUG agent_id after personalize: {agent.agent_id}")
            # Re-verify with retries to allow backend state to settle
            for _ in range(5):  # retry up to 5 times
                ok, report = agent.verify()
                if ok:
                    break
                print("[HelloWorld] ...waiting for agent to become ready...")
                time.sleep(1)

        if not ok:
            print("[HelloWorld] ❌ Agent not ready.")
            sys.exit(1)

    print("[HelloWorld] ✅ Agent personalized (instance registered in AOC) and verified.")

    # === Step 2: Prepare runtime once ===
    try:
        rt = agent.prepare_runtime()
        print(f"[HelloWorld] ✅ Runtime prepared")
        # Optional GGUF visibility:
        # lang_rt = (rt or {}).get("language") or {}
        # if lang_rt.get("gguf_path"):
        #     print(f"[HelloWorld] GGUF runtime detected: {lang_rt['gguf_path']}")
    except Exception as e:
        print(f"[HelloWorld] ❌ Runtime preparation failed: {e}")
        sys.exit(1)

    if os.getenv("HELLOWORLD_CI_ONESHOT", "0") == "1":
        prompt = os.getenv("HELLOWORLD_CI_PROMPT", "hello")
        try:
            result = agent.run(prompt, model_kind="language")
            reply = result.strip() if isinstance(result, str) else str(result)
            print(f"[HelloWorld][oneshot] prompt={prompt!r} reply={reply[:200]}")
            return
        except Exception as e:
            print(f"[HelloWorld][oneshot] failed: {e}")
            sys.exit(1)

    print("=== HelloWorld Chatbot ===")
    print("Type your message and press Enter. Type 'exit' to quit.\n")

    # ---- Interactive Loop ----
    while True:
        try:
            ok, report = agent.verify()
            if not ok:
                print("[HelloWorld] ❌ Agent verification failed (disabled/revoked). Sleeping...")
                time.sleep(5)
                continue

            # Prompt user for input
            user_input = input("\n👤 [You] > ").strip()
            if user_input.lower() in ("exit", "quit"):
                print("\n👋 [System] Exiting.")
                break
            if not user_input:
                continue

            # Send input to agent
            result = agent.run(user_input, model_kind="language")
            reply = result.strip() if isinstance(result, str) else str(result)

            print(f"\n🤖 [Agent] > {reply}")

            # Optional A2A send/inbox/ack (uncomment to enable):
            # if peer_agent:
            #     corr = f"{os.getenv('A2A_CORRELATION_PREFIX', 'hello')}-{int(time.time())}"
            #     sent = a2a.send_message(
            #         from_agent_id=agent.agent_id,
            #         to_agent_id=peer_agent,
            #         message_type="event",
            #         correlation_id=corr,
            #         payload={"prompt": user_input, "reply": reply},
            #     )
            #     print(f"[HelloWorld][A2A] sent message id={sent.get('message', {}).get('id')}")
            #
            # inbox = a2a.inbox(agent_id=agent.agent_id, limit=10)
            # for msg in inbox.get("items", []):
            #     print(f"[HelloWorld][A2A] inbox message from={msg.get('from_agent_id')} payload={msg.get('payload')}")
            #     a2a.ack_message(message_id=msg["id"], agent_id=agent.agent_id)

        except KeyboardInterrupt:
            print("\n[HelloWorld] Shutting down.")
            sys.exit(0)
        except Exception as e:
            print(f"\n[HelloWorld] Processing error: {e}")
            time.sleep(2)

if __name__ == "__main__":
    main()
