"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "AFK"
AGENT_NAME = "AFK-Adaptive"


# ===== YOUR CODE HERE =====

import random
import time

import strategy
from game import ObservedAction

# ---- Time budget -----------------------------------------------------------
TIME_BANK = 8.0
OVERHEAD_PER_DECISION = 0.0002   # assumed per-request cost charged to us (measured ~0.25ms < 1ms grace)
SAFE_RESERVE = 2.5               # seconds never spent on simulation (judge CPU may be slower)
POSTFLOP_DECISIONS_PER_HAND = 2.0
MIN_SAMPLES, MAX_SAMPLES = 24, 300


def new_half(config: AgentConfig, memory: dict) -> None:
    memory["hands_played"] = 0
    memory["time_used"] = 0.0
    memory["decisions"] = 0
    memory["caught_errors"] = 0
    memory["time_bank"] = getattr(config, "time_bank_seconds", TIME_BANK) or TIME_BANK
    memory["num_hands"] = getattr(config, "num_hands", 1000) or 1000
    memory["rng"] = random.Random(12345)
    memory["stats"] = strategy.new_stats()
    memory["info"] = strategy.new_info()
    memory["my_def"] = [0, 0, 0.0]     # our folds facing postflop bets, decisions, sum of break-even fold rates
    memory["calib_st"] = {1: [0.0, 0.0, 0], 2: [0.0, 0.0, 0], 3: [0.0, 0.0, 0]}   # per street
    memory["calib"] = [0.0, 0.0, 0]    # predicted equity, actual result, count (calls that reached showdown)
    memory["sample_cost"] = 60e-6      # seconds per MC sample, re-measured live
    memory["cur"] = None               # what we saw in the current hand


def _time_left(memory: dict) -> float:
    spent = memory.get("time_used", 0.0) + memory.get("decisions", 0) * OVERHEAD_PER_DECISION
    return memory.get("time_bank", TIME_BANK) - spent


def _n_samples(obs: Observation, memory: dict) -> int:
    hands_left = max(1, memory["num_hands"] - obs.hand_index)
    P = strategy.P
    dec = P["tb_dec"]
    if P["tb_adapt"] >= 0.5 and memory.get("hands_played", 0) >= 20:
        dec = max(dec, memory.get("post_decisions", 0) / max(1, memory["hands_played"]))
    budget = (_time_left(memory) - P["tb_reserve"]) / hands_left / dec
    n = int(budget / memory["sample_cost"])
    return max(MIN_SAMPLES, min(int(P["tb_max"]), n))


def _decide(obs: Observation, memory: dict) -> Action:
    if obs.street == Street.PREFLOP:
        return strategy.decide(obs, memory, 0)
    memory["post_decisions"] = memory.get("post_decisions", 0) + 1
    n = _n_samples(obs, memory)
    t0 = time.perf_counter()
    action = strategy.decide(obs, memory, n)
    dt = time.perf_counter() - t0
    used = memory.get("n_used") or n
    memory["sample_cost"] = 0.8 * memory["sample_cost"] + 0.2 * (dt / used)
    return action


def _safe_fallback(obs: Observation) -> Action:
    if Action.CHECK_CALL in obs.legal_actions:
        return Action.CHECK_CALL
    return obs.legal_actions[0]


def _cheap_policy(obs: Observation, memory: dict) -> Action:
    """Time bank nearly empty: preflop table + pot odds only, no simulation."""
    try:
        if obs.street == Street.PREFLOP:
            return strategy.decide(obs, memory, 0)
        if obs.to_call == 0:
            return Action.CHECK_CALL
        pct = strategy.pf_pct(obs.hole_cards)
        price = obs.to_call / (obs.pot + obs.to_call)
        return Action.CHECK_CALL if pct > 0.5 and price < 0.35 else Action.FOLD
    except Exception:
        return _safe_fallback(obs)


def choose_action(obs: Observation, memory: dict) -> Action:
    """Wrapper: time it, never crash, never return an illegal action."""
    t0 = time.perf_counter()
    try:
        if _time_left(memory) < min(SAFE_RESERVE, strategy.P["tb_reserve"]):
            action = _cheap_policy(obs, memory)
        else:
            action = _decide(obs, memory)
        if action not in obs.legal_actions:
            action = _safe_fallback(obs)
    except Exception:
        memory["caught_errors"] = memory.get("caught_errors", 0) + 1
        try:
            action = _safe_fallback(obs)
        except Exception:
            action = Action.CHECK_CALL
    try:
        prev = memory.get("cur")
        same = prev and prev.get("hand") == obs.hand_index
        last_call = prev.get("last_call") if same else None
        calls = list(prev.get("calls", [])) if same else []
        if (obs.street != Street.PREFLOP and obs.to_call > 0 and action == Action.CHECK_CALL
                and memory.get("call_pred") is not None):
            last_call = memory["call_pred"]
            calls.append((int(obs.street), memory["call_pred"]))
        memory["call_pred"] = None
        if obs.street != Street.PREFLOP and obs.to_call > 0 and obs.my_stack > 0:
            d = memory["my_def"]
            d[0] += 1 if action == Action.FOLD else 0
            d[1] += 1
            d[2] += obs.to_call / max(1, obs.pot)   # their bluff breaks even when we fold B/(P+B)
        memory["cur"] = {"hand": obs.hand_index, "last_call": last_call, "calls": calls, "hist": obs.action_history,
                         "button": obs.is_button, "street": obs.street,
                         "action": action, "to_call": obs.to_call}
    except Exception:
        pass
    memory["decisions"] = memory.get("decisions", 0) + 1
    memory["time_used"] = memory.get("time_used", 0.0) + (time.perf_counter() - t0)
    return action


def _learn(result: HandResult, memory: dict) -> None:
    """Rebuild the hand's action list and update opponent stats."""
    stats = memory["stats"]
    cur = memory.get("cur")
    if cur is None or cur["hand"] != result.hand_index:
        # We never acted: the opponent (small blind) folded straight away.
        if result.my_net > 0:
            strategy.record_opp(stats, Street.PREFLOP, Action.FOLD, False)
        return
    st = cur["street"]
    hist = list(cur["hist"])
    my_act = cur["action"]
    hist.append(ObservedAction(st, True, my_act, 0, -1))  # street_total unknown; not needed for opp
    # walk_history needs street totals for "faced" -> compute opp actions from the real history only
    for is_me, street, action, faced in strategy.walk_history(cur["hist"], cur["button"]):
        if not is_me:
            strategy.record_opp(stats, street, action, faced)
    if result.went_to_showdown and cur.get("last_call") is not None:
        actual = 1.0 if result.winner == "me" else (0.5 if result.winner == "split" else 0.0)
        c = memory["calib"]
        c[0] += cur["last_call"]; c[1] += actual; c[2] += 1
        for st, pred in cur.get("calls", []):
            cs = memory["calib_st"].setdefault(st, [0.0, 0.0, 0])
            cs[0] += pred; cs[1] += actual; cs[2] += 1
    if result.went_to_showdown and result.opp_hole:
        strategy.learn_showdown(memory["info"], stats, cur["hist"], cur["button"],
                                result.opp_hole, result.board, memory["rng"])
    # Infer the opponent's final action (after our last decision).
    i_raised = my_act in (Action.RAISE_HALF, Action.RAISE_POT)
    if my_act == Action.FOLD:
        return
    if not result.went_to_showdown and result.winner == "me":
        strategy.record_opp(stats, st, Action.FOLD, True)          # folded to our bet
    elif result.went_to_showdown and i_raised:
        strategy.record_opp(stats, st, Action.CHECK_CALL, True)    # called our bet
    elif (result.went_to_showdown and st == Street.RIVER and not cur["button"]
          and my_act == Action.CHECK_CALL and cur["to_call"] == 0):
        strategy.record_opp(stats, st, Action.CHECK_CALL, False)   # checked behind


def on_hand_end(result: HandResult, memory: dict) -> None:
    t0 = time.perf_counter()
    try:
        memory["hands_played"] = memory.get("hands_played", 0) + 1
        _learn(result, memory)
    except Exception:
        memory["caught_errors"] = memory.get("caught_errors", 0) + 1
    memory["time_used"] = memory.get("time_used", 0.0) + (time.perf_counter() - t0)

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
