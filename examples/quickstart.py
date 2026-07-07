"""agentpause quickstart — runnable with no API keys and no real provider.

It uses a tiny simulated backend and a shrinking rate-limit window so you can
watch the scheduler suspend *before* the budget runs out, then resume from the
exact step on a second run.

Run it twice:

    python examples/quickstart.py      # runs, then suspends mid-task
    python examples/quickstart.py      # resumes from where it stopped

(Delete the .agentpause/ folder to start over.)
"""

import random

from agentpause import Estimator, PredictiveScheduler

random.seed(7)  # deterministic demo

QUESTIONS = [
    "Summarize the risks of an LLM agent without resource management.",
    "Pick the most urgent risk and justify it.",
    "Propose a mitigation for that risk.",
    "What role does checkpointing play here?",
    "Give a two-sentence conclusion.",
]


# --- a simulated LLM backend: returns a canned reply and a random token cost ---
def fake_backend(messages):
    reply = f"[demo reply to: {messages[-1]['content'][:40]}...]"
    tokens = random.randint(260, 380)
    return reply, tokens


# --- a simulated rate-limit window that drains as the task proceeds ---
class DrainingWindow:
    def __init__(self, start=1400):
        self.remaining = start

    def __call__(self):
        # each check consumes a slice of the window, so a suspend happens partway
        self.remaining = max(0, self.remaining - random.randint(180, 260))
        return self.remaining


def main():
    # a light estimator matched to the small demo replies (~320 tokens/step)
    sched = PredictiveScheduler(
        backend=fake_backend,
        telemetry=DrainingWindow(1400),
        estimator=Estimator(max_tokens=350, tool_overhead=0),
        safety_k=2.0,
    )

    with sched.session("quickstart-demo") as s:
        if s.resumed:
            print(f"↻ Resuming from step {s.step + 1}\n")

        # skip questions already answered before a previous suspend
        for question in QUESTIONS[s.step:]:
            s.add_user(question)

            if s.should_suspend():
                s.checkpoint()
                print(f"🛑 Suspended before step {s.step + 1} "
                      f"(budget too low). State saved — run again to resume.")
                return

            reply = s.call()
            print(f"✓ step {s.step}: {reply}")

        s.complete()
        print(f"\n✅ Task complete in {s.step} steps, "
              f"{s.total_tokens_used} tokens used.")


if __name__ == "__main__":
    main()
