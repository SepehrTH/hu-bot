"""
strategy.py - the brain of the bot (imported by agent.py).

Plan:
  * Preflop: lookup tables built offline (offline/preflop.c -> preflop.bin):
    every 5-card hand's strength percentile and its all-in equity vs a
    random hand and vs the top 50% / 30% / 15% of hands.
  * Postflop: Monte Carlo equity vs the opponent's estimated RANGE.
  * Range = "top q" of hands, where q comes from how often THIS opponent
    takes the action they took (learned during the half). A bot that
    raises 100% of the time has q=1 (random range); a nit's raise means a
    narrow range.
  * Decision = equity vs price (pot odds), plus value bets, plus bluffs
    only against opponents measured to fold a lot.
"""
from __future__ import annotations

import os
import random
from array import array
from itertools import combinations
from math import comb

from evaluator import get_table
from game import Action, Street

# ---------------------------------------------------------------------------
# Tunable parameters (tools/tune.py can override these)
# ---------------------------------------------------------------------------
P = dict(
    sb_raise=0.78,     # SB opens with pot raise at/above this strength percentile
    sb_call=0.12,      # SB limps at/above this, folds below
    bb_raise=0.80,     # BB raises a limp at/above this percentile
    pf_margin=0.04,    # extra equity needed over pot odds to call preflop
    pf_reraise=0.58,   # 3-bet if equity vs their range above this
    post_margin=0.062, # extra equity over pot odds to call postflop
    post_raise=0.70,   # raise a bet if equity vs range above this
    bet_value=0.62,    # bet when checked to if equity above this
    bet_pot=0.72,      # ...pot-sized if above this, else half pot
    semi_bluff=0.447,  # bet half pot with this much equity if they fold enough
    semi_fold=0.40,    # ...needed fold-to-bet rate for semi-bluffs
    bluff_fold=0.55,   # pure bluffs only if fold-to-bet rate above this
    bluff_freq=0.5,    # how often to bluff in that case
    q_floor_pf=0.06,   # narrowest preflop range we ever assume
    q_floor_post=0.10, # narrowest postflop range we ever assume
    prior_n=12.0,      # prior pseudo-counts for opponent stats
    thin_k=1.0,        # value-bet threshold drop per unit of fold rate below 40%
    steal_k=1.0,       # how much wider we open per unit of excess preflop fold rate
    beta_prior=0.25,   # prior share of "air" (random hands) in their betting range
    beta_k1=0.30,      # ...per barrel: prior air share of a 1st postflop bet,
    beta_k2=0.22,      # ...of a 2nd-street bet,
    beta_k3=0.15,      # ...and of a 3rd (river after flop+turn) bet
    barrel_on=0.0,     # 1 = learn air share separately by barrel count (tested: worse vs top bots)
    beta_n=6.0,        # showdowns worth of that prior
    beta_floor=0.08,   # never assume less than this much air
    mdf_k=0.6,         # call-threshold drop per unit our fold rate exceeds the bluff break-even
    mdf_on=0.45,       # ...only when the opponent bets/raises at least this often when checked to
    mdf_max=0.15,      # largest threshold drop
    ev_bet=0.0,        # 1 = choose check / half / pot by expected value (v4); 0 = fixed thresholds (v3)
    ev_fold_k=0.9,     # how much bigger bets increase their fold rate
    ev_margin=0.02,    # a bet must beat checking by this fraction of the pot
    ev_raise_pen=0.5,  # penalty for being raised off the hand before the river
    calib_street=0.0,  # 1 = learn the equity correction separately for flop / turn / river
    bayes=0.0,         # 1 = v5 Bayesian range model (per-hand likelihoods), 0 = top-q + air mixture
    bz_k=10.0,         # steepness of the action likelihood curves
    bz_call_floor=0.25,# chance a hand below their folding threshold still calls (draws, floats)
    bz_check=0.5,      # how much a check lowers the weight of hands that would have bet
    bz_ess=12.0,       # below this effective sample size, blend toward unweighted equity
    bal_p=0.0,         # balanced bluffing: probability of betting a weak hand when checked to
    bal_e=0.30,        # ...hands below this equity count as bluff candidates
    bal_fold_min=0.30, # ...never against opponents who fold less than this to bets (stations)
    decay=1.0,         # opponent-stat recency: each new observation multiplies old counts by this (1 = off)
    tb_reserve=2.5,    # time bank: seconds never spent on simulation
    tb_dec=2.0,        # time bank: assumed postflop decisions per hand (lower = more samples each)
    tb_max=300,        # time bank: max Monte Carlo samples per decision
    tb_adapt=0.0,      # 1 = use the measured postflop decisions per hand (never assume fewer than tb_dec)

    trap_p=0.8,        # vs frequent stabbers: probability of checking a strong hand to check-raise
    trap_e=0.72,       # ...for hands with at least this equity
    trap_on=0.60,      # ...only if they bet at least this often when checked to
    close_band=0.06,   # facing a bet: if equity is this close to the call threshold, double the samples
    q_chain=0.5,       # earlier-street bets count only this much (exponent) toward range narrowing
    calib_prior=10.0,  # showdowns of prior belief that our equity estimates are unbiased
    calib_max=0.15,    # largest equity correction from calibration
    reraise_k=0.6,     # 3-bet threshold drop per unit of their fold-to-raise above 35%
    info_prior=4.0,    # showdowns worth of prior belief in "raises mean strength"
    info_w0=0.8,       # that prior belief (1 = fully informative)
)

try:  # testing hook only: BOT_PARAMS='{"key": value}' (never set in the tournament)
    import json as _json
    if os.environ.get("BOT_PARAMS"):
        P.update(_json.loads(os.environ["BOT_PARAMS"]))
except Exception:
    pass

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
_T = get_table()                       # mask -> score
BIT = [1 << c for c in range(36)]
PAIRS = tuple(combinations(range(5), 2))
_C = [[comb(n, k) for n in range(37)] for k in range(6)]   # _C[k][n]
C1, C2, C3, C4, C5 = _C[1], _C[2], _C[3], _C[4], _C[5]
NH = comb(36, 5)

_here = os.path.dirname(os.path.abspath(__file__))
_raw = array("H")
with open(os.path.join(_here, "preflop.bin"), "rb") as f:
    _raw.frombytes(f.read())
PCT = _raw[0:NH]
EQ = [_raw[NH * (i + 1):NH * (i + 2)] for i in range(4)]     # q = 1, .5, .3, .15
QS = (1.0, 0.5, 0.3, 0.15)
SCALE = 1.0 / 65535.0


def hidx(cards) -> int:
    a, b, c, d, e = sorted(cards)
    return C1[a] + C2[b] + C3[c] + C4[d] + C5[e]


def pf_pct(cards) -> float:
    return PCT[hidx(cards)] * SCALE


def pf_equity(cards, q: float) -> float:
    """All-in equity vs the top-q fraction of hands (interpolated)."""
    i = hidx(cards)
    v = [EQ[k][i] * SCALE for k in range(4)]
    if q >= 1.0:
        return v[0]
    for k in range(3):
        hi, lo = QS[k], QS[k + 1]
        if q >= lo:
            t = (q - lo) / (hi - lo)
            return v[k + 1] + t * (v[k] - v[k + 1])
    # below 0.15: extrapolate along the last segment
    slope = (v[2] - v[3]) / (0.3 - 0.15)
    return max(0.0, v[3] - slope * (0.15 - q))


# ---------------------------------------------------------------------------
# Opponent model
# ---------------------------------------------------------------------------
# contexts: (street_group, faced_bet) ; street_group 0 = preflop, 1 = postflop
def new_stats():
    return {ctx: {"n": 0, "raise": 0, "call": 0, "fold": 0}
            for ctx in ((0, False), (0, True), (1, False), (1, True))}


PRIOR = {  # default beliefs about an unknown opponent: (raise, call, fold) rates
    (0, False): (0.30, 0.70, 0.00),
    (0, True): (0.20, 0.50, 0.30),
    (1, False): (0.35, 0.65, 0.00),
    (1, True): (0.15, 0.45, 0.40),
}


def rate(stats, ctx, kind):
    s = stats[ctx]
    k = P["prior_n"]
    pr = PRIOR[ctx][("raise", "call", "fold").index(kind)]
    return (s[kind] + k * pr) / (s["n"] + k)


def walk_history(history, i_am_button):
    """Yield (is_me, street, action, faced_bet) for each action in a hand."""
    my_bet, opp_bet = (1, 2) if i_am_button else (2, 1)
    street = Street.PREFLOP
    for a in history:
        if a.street != street:
            street = a.street
            my_bet = opp_bet = 0
        # preflop: "faced" means facing a RAISE (the big blind doesn't count)
        floor = 2 if street == Street.PREFLOP else 0
        if a.is_me:
            faced = opp_bet > my_bet and opp_bet > floor
            my_bet = a.street_total
        else:
            faced = my_bet > opp_bet and my_bet > floor
            opp_bet = a.street_total
        yield a.is_me, a.street, a.action, faced


def record_opp(stats, street, action, faced):
    ctx = (0 if street == Street.PREFLOP else 1, faced)
    s = stats[ctx]
    d = P["decay"]
    if d < 1.0:   # recency: older observations count a little less, so we track adapting opponents
        for key in ("n", "raise", "call", "fold"):
            s[key] *= d
    s["n"] += 1
    if action == Action.FOLD:
        s["fold"] += 1
    elif action == Action.CHECK_CALL:
        s["call"] += 1
    else:
        s["raise"] += 1


def new_info():
    """Showdown evidence of how much the opponent's raises reveal.
    For each group (0 preflop, 1 postflop): sum of (observed strength - 0.5)
    and sum of (strength expected if raises were pure top-q - 0.5)."""
    return {0: [0.0, 0.0, 0], 1: [0.0, 0.0, 0], "air": [0, 0]}


def air_share(info, k=0):
    """Estimated fraction of the opponent's postflop bets made with weak hands.
    k = how many postflop streets they have bet so far this hand (1, 2, 3): first bets
    are often bluffs while third barrels are often value, so each is learned separately."""
    if k and P["barrel_on"] >= 0.5:
        prior = P["beta_k%d" % min(k, 3)]
        weak, n = (info or {}).get("air_k", {}).get(min(k, 3), (0, 0))
    else:
        prior = P["beta_prior"]
        weak, n = (info or {}).get("air", (0, 0))
    b = (weak + prior * P["beta_n"]) / (n + P["beta_n"])
    return max(P["beta_floor"], min(0.9, b))


def opp_barrels(history):
    """Number of distinct postflop streets on which the opponent bet or raised this hand."""
    return len({a.street for a in history if (not a.is_me) and a.street != Street.PREFLOP
                and a.action in (Action.RAISE_HALF, Action.RAISE_POT)})


def info_weight(info, g):
    """1 = their raises mean exactly what the frequency says, 0 = pure noise."""
    obs, exp, n = info[g]
    k = P["info_prior"]
    w = (obs + k * P["info_w0"] * 0.2) / (exp + k * 0.2)
    return max(0.0, min(1.0, w))


def opp_ranges(stats, history, i_am_button, info=None):
    """Return (q_pf, q_post): fraction of hands the opponent's line represents."""
    q_pf = 1.0
    post_f = []
    w = (info_weight(info, 0), info_weight(info, 1)) if info else (1.0, 1.0)
    for is_me, street, action, faced in walk_history(history, i_am_button):
        if is_me:
            continue
        g = 0 if street == Street.PREFLOP else 1
        ctx = (g, faced)
        if action in (Action.RAISE_HALF, Action.RAISE_POT):
            f = max(0.05, rate(stats, ctx, "raise"))
        elif action == Action.CHECK_CALL:
            f = max(0.25, 1.0 - rate(stats, ctx, "fold"))
        else:
            continue
        f = f ** w[g]
        if g == 0:
            q_pf *= f
        else:
            post_f.append(f)
    # Bets on consecutive streets are correlated (strong hands bet every street),
    # so only the latest one counts fully; earlier ones are damped.
    q_post = 1.0
    if post_f:
        q_post = post_f[-1]
        for f in post_f[:-1]:
            q_post *= f ** P["q_chain"]
    return max(P["q_floor_pf"], min(1.0, q_pf)), max(P["q_floor_post"], min(1.0, q_post))


# ---------------------------------------------------------------------------
# Monte Carlo equity vs range
# ---------------------------------------------------------------------------
def _best(pmasks, board_masks):
    T = _T
    best = 0
    for h in pmasks:
        for b in board_masks:
            s = T[h | b]
            if s > best:
                best = s
    return best


def _trip_masks(board):
    return [BIT[board[i]] | BIT[board[j]] | BIT[board[k]] for i, j, k in combinations(range(len(board)), 3)]


def _pair_masks(cards):
    return [BIT[cards[i]] | BIT[cards[j]] for i, j in PAIRS]


def mc_equity(hole, board, q_pf, q_post, n, rng, beta=0.0):
    """Equity vs a soft range: (1-beta) * [top-q_pf preflop, top-q_post by current
    made hand] + beta * random hands. beta keeps bluff-heavy lines from looking
    like the nuts."""
    dead = set(hole) | set(board)
    deck = [c for c in range(36) if c not in dead]
    my_p = _pair_masks(hole)
    nb = len(board)
    need = 5 - nb
    filt = q_pf < 0.999 or q_post < 0.999
    if not filt:
        beta = 0.0

    if need == 0:
        tm = _trip_masks(board)
        mine = _best(my_p, tm)
        if not filt:
            win = 0.0
            for _ in range(n):
                sc = _best(_pair_masks(rng.sample(deck, 5)), tm)
                win += 1.0 if mine > sc else (0.5 if mine == sc else 0.0)
            return win / n
        K = int(min(max(n, n / (q_pf * q_post)), 4 * n, 600))
        cands = [rng.sample(deck, 5) for _ in range(K)]
        scored = [(_best(_pair_masks(o), tm), o) for o in cands]
        def eq(lst):
            if not lst:
                return 0.5
            w = 0.0
            for sc, _ in lst:
                w += 1.0 if mine > sc else (0.5 if mine == sc else 0.0)
            return w / len(lst)
        e_rand = eq(scored)
        f = scored
        if q_pf < 0.999:
            f = sorted(f, key=lambda t: PCT[hidx(t[1])], reverse=True)
            f = f[:max(8, int(len(f) * q_pf + 0.5))]
        if q_post < 0.999:
            f = sorted(f, key=lambda t: t[0], reverse=True)
            f = f[:max(8, int(len(f) * q_post + 0.5))]
        return (1.0 - beta) * eq(f) + beta * e_rand

    opps = None
    if filt:
        K = int(min(max(n, n / (q_pf * q_post)), 4 * n, 600))
        cands = [rng.sample(deck, 5) for _ in range(K)]
        if q_pf < 0.999:
            cands.sort(key=lambda o: PCT[hidx(o)], reverse=True)
            cands = cands[:max(8, int(len(cands) * q_pf + 0.5))]
        if q_post < 0.999 and nb >= 3:
            tm = _trip_masks(board)
            cands.sort(key=lambda o: _best(_pair_masks(o), tm), reverse=True)
            cands = cands[:max(8, int(len(cands) * q_post + 0.5))]
        opps = cands

    def rollout(o):
        rest = [c for c in deck if c not in o]
        full = list(board) + rng.sample(rest, need)
        tm = _trip_masks(full)
        a = _best(my_p, tm)
        b = _best(_pair_masks(o), tm)
        return 1.0 if a > b else (0.5 if a == b else 0.0)

    if opps is None:
        win = 0.0
        for _ in range(n):
            win += rollout(rng.sample(deck, 5))
        return win / n
    n_r = int(n * beta + 0.5) if beta > 0 else 0
    n_f = max(1, n - n_r)
    e_f = sum(rollout(opps[i % len(opps)]) for i in range(n_f)) / n_f
    if n_r == 0:
        return e_f
    e_r = sum(rollout(rng.sample(deck, 5)) for _ in range(n_r)) / n_r
    return (1.0 - beta) * e_f + beta * e_r


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------
def _raise_or(obs, kind, fallback=Action.CHECK_CALL):
    return kind if kind in obs.legal_actions else fallback


def decide(obs, memory, n_samples):
    rng = memory["rng"]
    stats = memory["stats"]
    q_pf, q_post = opp_ranges(stats, obs.action_history, obs.is_button, memory.get("info"))
    can_raise = Action.RAISE_POT in obs.legal_actions
    to_call, pot = obs.to_call, obs.pot
    price = to_call / (pot + to_call) if to_call > 0 else 0.0

    # ------------------------------ preflop ------------------------------
    if obs.street == Street.PREFLOP:
        pct = pf_pct(obs.hole_cards)
        opp_raised = any((not a.is_me) and a.action in (Action.RAISE_HALF, Action.RAISE_POT)
                         for a in obs.action_history)
        if not opp_raised:
            # steal wider vs opponents that fold to preflop raises a lot
            pf_fold = rate(stats, (0, True), "fold")
            steal = P["steal_k"] * max(0.0, pf_fold - 0.35)
            if obs.is_button and obs.raises_this_street == 0:   # SB first action
                if pct >= P["sb_raise"] - steal and can_raise:
                    return Action.RAISE_POT
                if pct >= P["sb_call"]:
                    return Action.CHECK_CALL
                return Action.FOLD
            if to_call == 0:                                    # BB vs limp
                if pct >= P["bb_raise"] - steal and can_raise:
                    return Action.RAISE_POT
                return Action.CHECK_CALL
        # facing a raise (or our raise got called back to us)
        if to_call == 0:
            return Action.CHECK_CALL
        e = pf_equity(obs.hole_cards, q_pf)
        fold_vs = rate(stats, (0, True), "fold")
        reraise = P["pf_reraise"] - P["reraise_k"] * max(0.0, fold_vs - 0.35)
        if e >= reraise and can_raise:
            return Action.RAISE_POT
        if e >= price + P["pf_margin"]:
            return Action.CHECK_CALL
        return Action.FOLD

    # ------------------------------ postflop -----------------------------
    barrels = opp_barrels(obs.action_history)
    beta = air_share(memory.get("info"), barrels)
    if P["bayes"] >= 0.5:
        _eq = lambda: mc_equity_bayes(obs.hole_cards, obs.board, obs.action_history, obs.is_button,
                                      stats, memory.get("info"), n_samples, rng)
    else:
        _eq = lambda: mc_equity(obs.hole_cards, obs.board, q_pf, q_post, n_samples, rng, beta)
    e = _eq()
    memory["n_used"] = n_samples
    if to_call > 0:
        # Close call? Spend a second batch of samples to cut Monte Carlo noise where it matters.
        shift = calib_shift(memory["calib"], int(obs.street), memory.get("calib_st")) if "calib" in memory else 0.0
        if abs(e + shift - (price + P["post_margin"])) < P["close_band"]:
            e = 0.5 * (e + _eq())
            memory["n_used"] = 2 * n_samples
    memory["last_equity"] = e
    fold_rate = rate(stats, (1, True), "fold")
    if os.environ.get("BOT_DEBUG"):   # testing hook only (never set in the tournament)
        try:
            with open(os.environ["BOT_DEBUG"], "a") as fh:
                fh.write(_json.dumps({"h": obs.hand_index, "st": int(obs.street), "e": round(e, 3),
                    "sh": round(calib_shift(memory["calib"]), 3), "cal": memory["calib"][2],
                    "qpf": round(q_pf, 3), "qpo": round(q_post, 3), "tc": to_call, "pot": pot,
                    "air": round(air_share(memory.get("info")), 3),
                    "hole": list(obs.hole_cards), "board": list(obs.board)}) + "\n")
        except Exception:
            pass
    if to_call > 0:
        e_raw = e
        e = e + calib_shift(memory["calib"], int(obs.street), memory.get("calib_st")) if "calib" in memory else e
        memory["call_pred"] = e_raw
        if e >= P["post_raise"] and can_raise:
            return Action.RAISE_POT
        # Defence controller: if we fold more often than their bets need to
        # break even, and they bet a lot, call wider until bluffing stops paying.
        relief = 0.0
        my = memory.get("my_def")
        if my and rate(stats, (1, False), "raise") >= P["mdf_on"] and (barrels <= 2 or P["barrel_on"] < 0.5):
            folds, n, need = my
            if n >= 15:
                excess = folds / n - need / n
                relief = min(P["mdf_max"], max(0.0, P["mdf_k"] * excess))
        if e >= price + P["post_margin"] - relief:
            return Action.CHECK_CALL
        return Action.FOLD
    # checked to us / first to act
    # Trap: against opponents who bet most of the time when checked to, sometimes check a
    # strong hand so they bet into it and we raise (post_raise). Only when we act first,
    # otherwise checking just ends the street.
    if (P["trap_p"] > 0 and can_raise and e >= P["trap_e"] and not obs.is_button
            and obs.street != Street.RIVER
            and rate(stats, (1, False), "raise") >= P["trap_on"] and rng.random() < P["trap_p"]):
        return Action.CHECK_CALL
    if P["ev_bet"] >= 0.5 and can_raise:
        return _ev_bet(obs, memory, e, q_pf, q_post, n_samples, rng, fold_rate)
    # thinner value bets vs opponents that rarely fold (they pay off worse hands)
    bet_value = max(0.52, P["bet_value"] - P["thin_k"] * max(0.0, 0.40 - fold_rate))
    if e >= bet_value and can_raise:
        return Action.RAISE_POT if e >= P["bet_pot"] else _raise_or(obs, Action.RAISE_HALF)
    if can_raise and e >= P["semi_bluff"] and fold_rate >= P["semi_fold"]:
        return Action.RAISE_HALF
    # Balanced bluffing: a fixed small share of weak hands bet half pot, so our bets don't
    # automatically mean strength to opponents that model us. Not vs calling stations.
    if (P["bal_p"] > 0 and can_raise and e < P["bal_e"] and fold_rate >= P["bal_fold_min"]
            and rng.random() < P["bal_p"]):
        return _raise_or(obs, Action.RAISE_HALF)
    bluff_p = P["bluff_freq"] + (1.0 - P["bluff_freq"]) * min(1.0, max(0.0, (fold_rate - P["bluff_fold"]) / 0.15))
    if can_raise and fold_rate >= P["bluff_fold"] and rng.random() < bluff_p:
        return Action.RAISE_HALF
    return Action.CHECK_CALL


def learn_showdown(info, stats, history, i_am_button, opp_hole, board, rng):
    """After a showdown: did the opponent's raises come with strong hands?"""
    raised = {0: False, 1: False}
    for is_me, street, action, faced in walk_history(history, i_am_button):
        if not is_me and action in (Action.RAISE_HALF, Action.RAISE_POT):
            raised[0 if street == Street.PREFLOP else 1] = True
    if raised[0]:
        q = max(0.05, rate(stats, (0, False), "raise") * 0.5 + rate(stats, (0, True), "raise") * 0.5)
        info[0][0] += pf_pct(opp_hole) - 0.5
        info[0][1] += (1 - q / 2) - 0.5
        info[0][2] += 1
    if raised[1] and len(board) == 5:
        q = max(0.05, rate(stats, (1, False), "raise") * 0.5 + rate(stats, (1, True), "raise") * 0.5)
        e = mc_equity(opp_hole, board, 1.0, 1.0, 40, rng)
        air = info.setdefault("air", [0, 0])
        air[0] += 1 if e < 0.5 else 0
        air[1] += 1
        k = min(3, opp_barrels(history))
        if k:
            ak = info.setdefault("air_k", {}).setdefault(k, [0, 0])
            ak[0] += 1 if e < 0.5 else 0
            ak[1] += 1
        info[1][0] += e - 0.5
        info[1][1] += (1 - q / 2) - 0.5
        info[1][2] += 1


def calib_shift(calib, street=None, calib_st=None):
    """Equity correction learned from showdowns after we called a bet (per street if enabled)."""
    if P["calib_street"] >= 0.5 and calib_st is not None and street is not None and street in calib_st:
        calib = calib_st[street]
    pred, act, n = calib
    k = P["calib_prior"]
    shift = (act - pred) / (n + k)
    m = P["calib_max"]
    return max(-m, min(m, shift))


def _ev_bet(obs, memory, e, q_pf, q_post, n, rng, fold_rate):
    """v4: pick check / half pot / pot by expected value.
    A bet of B into pot P: they fold with prob f (their measured fold rate, higher for bigger
    bets); when they continue, they hold the top (1-f) of their current range, so our equity
    there is e_c. EV(bet) = f*P + (1-f)*(e_c*(P + 2B) - B). EV(check) ~ e*P."""
    P_ = obs.pot
    beta = air_share(memory.get("info"))
    best_a, best_ev = Action.CHECK_CALL, e * P_
    nc = max(24, n // 2)
    used = memory.get("n_used", n)
    for a in (Action.RAISE_HALF, Action.RAISE_POT):
        if a not in obs.legal_actions or a not in obs.raise_to:
            continue
        B = obs.raise_to[a] - obs.my_bet
        if B <= 0:
            continue
        ratio = B / (P_ + B)
        f = min(0.95, max(0.0, fold_rate * (1.0 - P["ev_fold_k"] * (0.5 - ratio))))
        q_c = max(0.05, q_post * (1.0 - f))
        e_c = mc_equity(obs.hole_cards, obs.board, q_pf, q_c, nc, rng, beta * 0.5)
        used += nc
        ev = f * P_ + (1.0 - f) * (e_c * (P_ + 2 * B) - B)
        if obs.street != Street.RIVER:
            ev -= (1.0 - f) * B * max(0.0, 0.6 - e_c) * P["ev_raise_pen"]
        if ev > best_ev + P["ev_margin"] * P_:
            best_a, best_ev = a, ev
    memory["n_used"] = used
    return best_a


# ---------------------------------------------------------------------------
# v5: Bayesian range model
# ---------------------------------------------------------------------------
from bisect import bisect_left as _bl, bisect_right as _br
from math import exp as _exp


def _sig(x):
    if x > 30:
        return 1.0
    if x < -30:
        return 0.0
    return 1.0 / (1.0 + _exp(-x))


def _pcts(values):
    """Percentile of each value within the list (ties averaged)."""
    srt = sorted(values)
    n = float(len(srt))
    return [(_bl(srt, v) + _br(srt, v)) / (2.0 * n) for v in values]


def mc_equity_bayes(hole, board, history, i_am_button, stats, info, n, rng):
    """Equity vs the opponent's range, where every sampled opponent hand is weighted by the
    likelihood of all their actions this hand given that hand's strength at the time:
      raise  -> likely for hands above their (1 - raise rate) quantile, plus a bluff floor
      call   -> likely for hands above their fold-rate quantile, plus a floor for draws/floats
      check  -> strong hands become less likely (they would usually have bet)
    Strength = preflop percentile (preflop actions) or made-hand percentile on the board as
    it was on that street. Floors come from what showdowns taught us (air share, honesty)."""
    dead = set(hole) | set(board)
    deck = [c for c in range(36) if c not in dead]
    my_p = _pair_masks(hole)
    nb = len(board)
    need = 5 - nb
    cands = [rng.sample(deck, 5) for _ in range(n)]
    pm = [_pair_masks(o) for o in cands]
    acts = [(st, act, faced) for is_me, st, act, faced in walk_history(history, i_am_button) if not is_me]
    streets = sorted({int(st) for st, _, _ in acts if st != Street.PREFLOP and 2 + int(st) <= nb})
    strength = {}
    if any(st == Street.PREFLOP for st, _, _ in acts):
        strength[0] = _pcts([PCT[hidx(o)] for o in cands])
    for st in streets:
        tm = _trip_masks(board[:2 + st])
        strength[st] = _pcts([_best(p, tm) for p in pm])
    w_pf = info_weight(info, 0) if info else 1.0
    w_po = info_weight(info, 1) if info else 1.0
    beta = air_share(info)
    k = P["bz_k"]
    weights = [1.0] * n
    for st, act, faced in acts:
        g = 0 if st == Street.PREFLOP else 1
        s_list = strength.get(int(st))
        if s_list is None:
            continue
        ctx = (g, faced)
        if act in (Action.RAISE_HALF, Action.RAISE_POT):
            thr = 1.0 - max(0.03, rate(stats, ctx, "raise"))
            floor = 1.0 - (w_pf if g == 0 else w_po) * (1.0 - (0.08 if g == 0 else beta))
            for i in range(n):
                weights[i] *= floor + (1.0 - floor) * _sig(k * (s_list[i] - thr))
        elif act == Action.CHECK_CALL and (faced or (g == 0 and not i_am_button)):
            # a call (or the small blind's limp): their weakest hands would have folded
            thr = rate(stats, ctx, "fold")
            fl = P["bz_call_floor"]
            for i in range(n):
                weights[i] *= fl + (1.0 - fl) * _sig(k * (s_list[i] - thr))
        elif act == Action.CHECK_CALL:
            # a check (postflop, or the big blind checking our limp): strong hands would often have bet
            thr = 1.0 - max(0.03, rate(stats, (g, False), "raise"))
            c = P["bz_check"] * (w_pf if g == 0 else w_po)
            for i in range(n):
                weights[i] *= 1.0 - c * _sig(k * (s_list[i] - thr))
    # outcomes
    if need == 0:
        tm = _trip_masks(board)
        mine = _best(my_p, tm)
        outs = []
        for p in pm:
            sc = _best(p, tm)
            outs.append(1.0 if mine > sc else (0.5 if mine == sc else 0.0))
    else:
        outs = []
        for o, p in zip(cands, pm):
            rest = [c for c in deck if c not in o]
            full = list(board) + rng.sample(rest, need)
            tm = _trip_masks(full)
            a, b = _best(my_p, tm), _best(p, tm)
            outs.append(1.0 if a > b else (0.5 if a == b else 0.0))
    sw = sum(weights)
    if sw <= 0:
        return sum(outs) / n
    e_w = sum(w * o for w, o in zip(weights, outs)) / sw
    ess = sw * sw / max(1e-12, sum(w * w for w in weights))
    if ess < P["bz_ess"]:
        t = ess / P["bz_ess"]
        e_w = t * e_w + (1.0 - t) * (sum(outs) / n)
    return e_w
