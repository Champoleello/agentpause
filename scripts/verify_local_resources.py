"""Live verification of the local-resources budgets against a REAL llama-server.

This is a one-off diagnostic script, not a permanent example (see
examples/local_resources_quickstart.py for the offline, fake-based demo).
Its whole point is to answer the one question the code itself couldn't
answer from documentation alone: what does a REAL `GET /slots` response
from your llama-server actually look like, and does our best-guess
`used_field` chain (cache_tokens -> n_past -> prompt -> tokens ->
next_token.n_decoded) land on the right one?

Run against the SAME server/model Andrea already used for the KV-cache
live test (see archivio/prototipi_root_luglio2026/avvia_server_8b.command):

    llama-server -hf unsloth/Qwen3-8B-GGUF:Q4_K_M \\
        --slot-save-path ./kv_cache -c 10240 --port 8080

Then, from the `agentpause` repo root, on the `feature/local-context-budget`
branch:

    python scripts/verify_local_resources.py

No GPU required for this part (Metal/CPU backends both expose the same
HTTP API) -- GPUMemoryBudget (pynvml/NVIDIA-only) is NOT exercised here on
purpose, since this machine has no NVIDIA GPU; that piece stays unverified
until tested on NVIDIA hardware.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import httpx

from agentpause.llamacpp_kv import LlamaCppSlots, KVStateStore
from agentpause.adapters.local_resources import LlamaCppContextBudget, KVAwareTimeBudget
from agentpause.state import StateStore
from agentpause.errors import KVError, TelemetryError

BASE_URL = "http://127.0.0.1:8080"
ID_SLOT = 0
KV_DIR = os.path.join(os.path.dirname(__file__), "..", "kv_cache")  # MUST match --slot-save-path


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_server() -> bool:
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        return r.status_code == 200
    except Exception as exc:
        print(f"Server non raggiungibile su {BASE_URL}: {exc}")
        print("Avvia prima il server (vedi docstring in cima a questo file).")
        return False


def main() -> None:
    section("0. Server check")
    if not check_server():
        sys.exit(1)
    print(f"OK, server raggiungibile su {BASE_URL}")

    slots = LlamaCppSlots()

    section("1. GET /props (raw)")
    props = slots.get_props(BASE_URL)
    print(json.dumps(props, indent=2)[:2000])
    dgs = props.get("default_generation_settings", {}) or {}
    n_ctx_reported = dgs.get("n_ctx") or props.get("n_ctx")
    print(f"\n--> n_ctx letto (assunzione del codice): {n_ctx_reported}")
    print("    CONFRONTA con -c 10240 passato al server: devono coincidere.")

    section("2. GET /slots (raw) -- PRIMA di generare qualunque testo")
    try:
        slot0_before = slots.get_slot(BASE_URL, ID_SLOT)
        print(json.dumps(slot0_before, indent=2)[:2000])
    except KVError as exc:
        print(f"get_slot ha fallito: {exc}")
        sys.exit(1)

    section("3. Genera un po' di contesto reale (POST /completion)")
    prompt = (
        "Elenca in una frase breve tre pianeti del sistema solare. "
        "Poi ripeti la stessa frase altre due volte con parole diverse. " * 3
    )
    print("Invio un prompt reale al modello per far crescere il KV-cache dello slot...")
    t0 = time.monotonic()
    resp = httpx.post(
        f"{BASE_URL}/completion",
        json={"prompt": prompt, "n_predict": 200, "id_slot": ID_SLOT, "cache_prompt": True},
        timeout=120.0,
    )
    resp.raise_for_status()
    body = resp.json()
    print(f"Generazione completata in {time.monotonic() - t0:.1f}s")
    print(f"Token generati (tokens_predicted, se riportato): {body.get('tokens_predicted')}")
    print(f"Token nel prompt (tokens_evaluated, se riportato): {body.get('tokens_evaluated')}")

    section("4. GET /slots (raw) -- DOPO aver generato testo")
    slot0_after = slots.get_slot(BASE_URL, ID_SLOT)
    print(json.dumps(slot0_after, indent=2)[:2000])
    print("\n--> CONFRONTA slot0_before vs slot0_after: quale campo è cambiato in")
    print("    modo coerente con la crescita del contesto? QUELLO è il campo giusto")
    print("    da passare esplicitamente come used_field= se il default indovinato")
    print("    sotto risulta sbagliato.")

    section("5. LlamaCppContextBudget -- cosa restituisce OGGI col default")
    budget_fn = LlamaCppContextBudget(slots, base_url=BASE_URL, id_slot=ID_SLOT, safety_margin_tokens=100)
    try:
        b = budget_fn()
        print(f"Budget: remaining_tokens={b.remaining_tokens}, limit_tokens={b.limit_tokens}")
        print("--> Il campo 'used' che ha vinto nella catena di tentativi è quello")
        print("    che il default_used_field ha trovato per primo (vedi punto 4:")
        print("    quale chiave era presente in slot0_after) -- confrontalo con quale")
        print("    campo hai visto CAMBIARE davvero al punto 4. Se non coincide,")
        print("    richiama questa classe passando used_field=<tua funzione>.")
    except TelemetryError as exc:
        print(f"TelemetryError (atteso se nessun campo noto è presente): {exc}")

    section("6. KVAwareTimeBudget -- riserva di tempo per il salvataggio KV")
    try:
        ktb = KVAwareTimeBudget(
            inner_telemetry=budget_fn,
            time_budget_s=600.0,
            estimated_kv_save_s=5.0,  # stima grezza, aggiusta dopo aver visto il tempo reale al punto 7
        )
        b2 = ktb()
        print(f"remaining_tokens={b2.remaining_tokens}, remaining_seconds={b2.remaining_seconds:.1f}")
    except TelemetryError as exc:
        print(f"Saltato (dipende dal punto 5, che ha fallito): {exc}")

    section("7. KV save/restore reale (KVStateStore) + guardia disco")
    os.makedirs(KV_DIR, exist_ok=True)
    store = KVStateStore(StateStore(os.path.join(os.path.dirname(__file__), "..", ".agentpause_verify")),
                         slots, BASE_URL, id_slot=ID_SLOT, kv_dir=KV_DIR)
    from agentpause.state import Checkpoint
    cp = Checkpoint(session_id="verify-local-resources")
    cp.messages.append({"role": "user", "content": prompt})
    cp.messages.append({"role": "assistant", "content": body.get("content", "")})

    t0 = time.monotonic()
    cp = store.save_with_kv(cp)
    save_s = time.monotonic() - t0
    kv_info = cp.extra["kv"]
    blob_path = os.path.join(KV_DIR, kv_info["file"])
    blob_bytes = os.path.getsize(blob_path) if os.path.exists(blob_path) else None
    print(f"Salvato in {save_s:.2f}s. n_saved={kv_info['n_saved']} celle, blob={blob_bytes} byte")
    if blob_bytes and kv_info["n_saved"]:
        print(f"--> bytes_per_token calibrato per questo modello: {blob_bytes / kv_info['n_saved']:.1f}")
        print("    (passa questo valore a GPUMemoryBudget(bytes_per_token=...) quando")
        print("     testerai quella classe su una macchina con GPU NVIDIA)")

    section("8. Guardia disco -- forziamo il fallimento apposta (min_free_bytes assurdo)")
    store_guarded = KVStateStore(
        StateStore(os.path.join(os.path.dirname(__file__), "..", ".agentpause_verify")),
        slots, BASE_URL, id_slot=ID_SLOT, kv_dir=KV_DIR,
        min_free_bytes=10 * 1024 ** 4,  # 10 TB: nessun disco reale ce l'ha, deve fallire
        prune_oldest=False,
    )
    cp2 = Checkpoint(session_id="verify-local-resources-guard-test")
    try:
        store_guarded.save_with_kv(cp2)
        print("ATTENZIONE: non ha sollevato KVError, controlla la logica della guardia.")
    except KVError as exc:
        print(f"OK, KVError sollevato come atteso: {exc}")

    section("9. Restore reale")
    loaded_cp, info = store.load_with_kv("verify-local-resources")
    print(f"kv_restored={info.get('kv_restored')}, dettagli={info}")

    section("RIEPILOGO -- incolla questa sezione in chat")
    print(f"n_ctx dal server: {n_ctx_reported}")
    print(f"Chiavi presenti in /slots PRIMA: {sorted(slot0_before.keys())}")
    print(f"Chiavi presenti in /slots DOPO:  {sorted(slot0_after.keys())}")
    print(f"Budget calcolato da LlamaCppContextBudget: remaining={b.remaining_tokens if 'b' in dir() else 'N/A'}")
    print(f"Tempo di salvataggio KV reale: {save_s:.2f}s (per calibrare estimated_kv_save_s)")
    print(f"n_saved={kv_info['n_saved']}, blob_bytes={blob_bytes}")


if __name__ == "__main__":
    main()
