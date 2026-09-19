"""Agentic AI layer: Supervisor -> Risk Agent -> Response Agent.

This sits on top of the Event Correlator. It does not run a second CV
model and does not call an LLM: it is a deterministic, explainable
decision layer over the structured events the existing detectors already
produce, matching the rest of this codebase's rule-based, auditable
design philosophy rather than bolting on an opaque black box.

  Structured Events -> Event Correlator -> Agentic Supervisor
                                              |-> Risk Agent      (assess)
                                              '-> Response Agent  (decide)

- RiskAgent:      transparent weighted scoring of an incident's events
                   into a 0-100 risk_score and a LOW/MEDIUM/HIGH/CRITICAL
                   risk_severity, with a human-readable explanation.
- ResponseAgent:  maps a risk assessment to a fixed table of pre-approved
                   response actions (alerting, evidence retention, human
                   verification, security notification). It never invents
                   new actions -- only ones from AGENTIC_BASE_ACTIONS.
- AgenticSupervisor: owns the EventCorrelator, feeds it events in real
                   time, and coordinates Risk + Response for both live
                   (in-progress) incidents and the final report.
"""
import math
from collections import defaultdict

from safety_config import (
    AGENTIC_RISK_WEIGHTS, AGENTIC_DEFAULT_EVENT_WEIGHT, AGENTIC_SEVERITY_MULTIPLIER,
    AGENTIC_MULTI_SIGNAL_BONUS, AGENTIC_WEAPON_CORRELATION_BONUS,
    AGENTIC_RISK_CRITICAL_THRESHOLD, AGENTIC_RISK_HIGH_THRESHOLD, AGENTIC_RISK_MEDIUM_THRESHOLD,
    AGENTIC_BASE_ACTIONS, AGENTIC_ACTION_LABELS,
)
from event_correlator import EventCorrelator


class RiskAgent:
    """Assesses the overall risk/severity of a correlated incident."""

    @staticmethod
    def _severity_multiplier(events):
        best = 1.0
        for e in events:
            sev = str(e.get("severity", "MEDIUM")).upper()
            best = max(best, AGENTIC_SEVERITY_MULTIPLIER.get(sev, 1.0))
        return best

    def assess(self, incident):
        events = incident.get("events", [])
        by_type = defaultdict(list)
        for e in events:
            by_type[e.get("event_type", "EVENT")].append(e)

        contributions = []
        raw_score = 0.0
        for event_type, evs in by_type.items():
            weight = AGENTIC_RISK_WEIGHTS.get(event_type, AGENTIC_DEFAULT_EVENT_WEIGHT)
            best_conf = max(float(e.get("confidence", 0.6) or 0.6) for e in evs)
            sev_mult = self._severity_multiplier(evs)
            # Diminishing returns for repeats of the same event type so a
            # noisy detector re-firing on one person can't dominate the score.
            count_mult = 1.0 + 0.15 * (math.sqrt(len(evs)) - 1.0)
            contribution = weight * sev_mult * best_conf * count_mult
            contributions.append({
                "event_type": event_type,
                "occurrences": len(evs),
                "contribution": round(contribution, 1),
            })
            raw_score += contribution

        bonuses = []
        distinct_types = len(by_type)
        if distinct_types >= 2:
            raw_score += AGENTIC_MULTI_SIGNAL_BONUS
            bonuses.append(
                f"+{AGENTIC_MULTI_SIGNAL_BONUS:.0f} multi-signal correlation "
                f"({distinct_types} independent event types in one incident)"
            )
        if "WEAPON_CONFIRMED" in by_type and distinct_types >= 2:
            raw_score += AGENTIC_WEAPON_CORRELATION_BONUS
            bonuses.append(
                f"+{AGENTIC_WEAPON_CORRELATION_BONUS:.0f} confirmed weapon co-occurring "
                f"with other risk signals"
            )

        score = max(0.0, min(100.0, raw_score))

        # A CONFIRMED weapon has already passed strict multi-frame,
        # multi-observation confirmation upstream (WeaponEventTracker), so
        # it is never allowed to score below CRITICAL on its own -- a real
        # armed threat should never be triaged down to "HIGH" just because
        # nothing else happened to correlate with it yet.
        weapon_floor = "WEAPON_CONFIRMED" in by_type
        if weapon_floor:
            score = max(score, AGENTIC_RISK_CRITICAL_THRESHOLD)

        if score >= AGENTIC_RISK_CRITICAL_THRESHOLD:
            severity = "CRITICAL"
        elif score >= AGENTIC_RISK_HIGH_THRESHOLD:
            severity = "HIGH"
        elif score >= AGENTIC_RISK_MEDIUM_THRESHOLD:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        if weapon_floor:
            bonuses.append("Risk floored to CRITICAL: a confirmed weapon is never triaged below CRITICAL")

        explanation = "; ".join(
            [f"{c['event_type']} x{c['occurrences']} (+{c['contribution']})" for c in contributions]
            + bonuses
        ) or "No contributing events."

        return {
            "risk_score": round(score, 1),
            "risk_severity": severity,
            "risk_factors": contributions,
            "risk_bonuses": bonuses,
            "risk_explanation": explanation,
        }


class ResponseAgent:
    """Determines the approved response actions for an assessed incident.

    Only ever returns actions from the fixed AGENTIC_BASE_ACTIONS table
    (plus the two conditional additions below) -- it does not synthesize
    arbitrary actions, so every possible output is something a human has
    already pre-approved.
    """

    def determine(self, risk_assessment, incident):
        severity = risk_assessment.get("risk_severity", "LOW")
        actions = list(AGENTIC_BASE_ACTIONS.get(severity, ["LOG_INCIDENT"]))
        event_types = set(incident.get("event_types", []))

        # An armed threat always requires a human to verify and security to
        # be told, even if the correlated incident otherwise scored lower.
        if "WEAPON_CONFIRMED" in event_types:
            for action in ("RETAIN_EVIDENCE", "NOTIFY_SECURITY_TEAM", "REQUIRE_HUMAN_VERIFICATION"):
                if action not in actions:
                    actions.append(action)

        # A possible fall always deserves at least a live alert, since a
        # person down can be time-critical even in an otherwise quiet scene.
        if "POSSIBLE_FALL" in event_types and "DISPATCH_ALERT" not in actions:
            actions.append("DISPATCH_ALERT")

        return {
            "response_actions": actions,
            "response_action_labels": [AGENTIC_ACTION_LABELS.get(a, a) for a in actions],
        }


class AgenticSupervisor:
    """Coordinates Event Correlation -> Risk Agent -> Response Agent.

    Usage:
      - ingest_event(event) each time a detector fires (real time, during
        frame processing) -> returns that event's fully-scored incident so
        the caller can react immediately (e.g. draw an on-frame banner).
      - active_incident_summaries() -> currently open (not yet closed)
        incidents, scored, useful for a "what's happening right now" view.
      - finalize() -> closes every remaining open incident, scores them
        all, and returns the complete incident report for the video.
    """

    def __init__(self):
        self.correlator = EventCorrelator()
        self.risk_agent = RiskAgent()
        self.response_agent = ResponseAgent()
        self.reset()

    def reset(self):
        self.correlator.reset()
        self.final_incidents = []

    def _score(self, incident_dict):
        risk = self.risk_agent.assess(incident_dict)
        response = self.response_agent.determine(risk, incident_dict)
        incident_dict.update(risk)
        incident_dict.update(response)
        return incident_dict

    def ingest_event(self, event):
        incident = self.correlator.ingest(event)
        return self._score(incident.to_dict())

    def active_incident_summaries(self):
        return [self._score(inc.to_dict()) for inc in self.correlator.active()]

    def finalize(self):
        incidents = self.correlator.finalize()
        final = [self._score(d) for d in incidents]
        final.sort(key=lambda i: i["start_time"])
        self.final_incidents = final
        return final
