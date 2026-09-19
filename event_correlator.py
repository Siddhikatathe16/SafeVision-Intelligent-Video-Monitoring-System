"""Event Correlation layer.

Converts the stream of independent structured events produced by the
existing CV detectors (crowd, loitering, restricted zone, abnormal
movement, fall) and the weapon tracker into *incidents*: groups of events
that are close together in time and share a subject (a specific tracked
person, or the whole scene for crowd-wide events).

This mirrors how a human operator actually reads a bank of CCTV alerts:
a crowd buildup, a sudden abnormal movement and a fall arriving within a
few seconds of each other in the same area is one unfolding situation,
not three unrelated alerts. Correlation is deliberately simple, rule
based and fully explainable (time-window clustering + subject overlap) —
no black-box model decides what counts as "related."
"""
from safety_config import AGENTIC_CORRELATION_WINDOW_SECONDS


class Incident:
    """A mutable, in-progress group of correlated structured events."""

    def __init__(self, incident_id, event):
        self.incident_id = incident_id
        self.events = [dict(event)]
        self.start_time = float(event.get("timestamp", 0.0))
        self.last_time = self.start_time
        self.person_ids = set()
        pid = event.get("person_id")
        if pid is not None:
            self.person_ids.add(int(pid))

    def matches(self, event, window):
        """Whether `event` belongs to this incident.

        Eligible if it arrives within `window` seconds of the incident's
        most recent event, AND either:
          - the event has no specific person (a scene-wide event such as
            HIGH_DENSITY) so it can attach to whatever is already unfolding, or
          - the incident has no person-specific events yet (it started from
            a scene-wide event, so any person-specific event happening now
            is plausibly part of the same situation), or
          - the event's person_id is already part of this incident.
        """
        if float(event.get("timestamp", 0.0)) - self.last_time > window:
            return False
        pid = event.get("person_id")
        if pid is None or not self.person_ids:
            return True
        return int(pid) in self.person_ids

    def add(self, event):
        self.events.append(dict(event))
        self.last_time = max(self.last_time, float(event.get("timestamp", 0.0)))
        pid = event.get("person_id")
        if pid is not None:
            self.person_ids.add(int(pid))

    def to_dict(self):
        event_types = sorted({e.get("event_type", "EVENT") for e in self.events})
        return {
            "incident_id": self.incident_id,
            "start_time": round(self.start_time, 2),
            "end_time": round(self.last_time, 2),
            "duration_seconds": round(max(0.0, self.last_time - self.start_time), 2),
            "person_ids": sorted(self.person_ids),
            "event_types": event_types,
            "event_count": len(self.events),
            "events": [dict(e) for e in self.events],
        }


class EventCorrelator:
    """Streaming clustering of structured events into incidents."""

    def __init__(self, window_seconds=None):
        self.window = float(window_seconds or AGENTIC_CORRELATION_WINDOW_SECONDS)
        self.reset()

    def reset(self):
        self.open_incidents = []
        self.closed_incidents = []
        self._counter = 0

    def _new_id(self):
        self._counter += 1
        return f"INC-{self._counter:04d}"

    def _close_stale(self, now):
        still_open = []
        for inc in self.open_incidents:
            if now - inc.last_time > self.window:
                self.closed_incidents.append(inc)
            else:
                still_open.append(inc)
        self.open_incidents = still_open

    def ingest(self, event):
        """Feed one structured event. Returns the Incident it joined/started."""
        now = float(event.get("timestamp", 0.0))
        self._close_stale(now)

        best = None
        for inc in self.open_incidents:
            if not inc.matches(event, self.window):
                continue
            pid = event.get("person_id")
            # Prefer an incident that already has this exact person over one
            # that would only match via the "scene-wide" fallback rule.
            if best is None:
                best = inc
            elif pid is not None and int(pid) in inc.person_ids and int(pid) not in best.person_ids:
                best = inc

        if best is None:
            best = Incident(self._new_id(), event)
            self.open_incidents.append(best)
        else:
            best.add(event)
        return best

    def active(self):
        """Open incidents that have not yet closed (no dict conversion)."""
        return list(self.open_incidents)

    def finalize(self):
        """Close every remaining open incident and return all incidents as dicts."""
        self.closed_incidents.extend(self.open_incidents)
        self.open_incidents = []
        ordered = sorted(self.closed_incidents, key=lambda i: i.start_time)
        return [inc.to_dict() for inc in ordered]
