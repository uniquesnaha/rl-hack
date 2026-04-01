"""
DSAR Compliance Environment — GDPR data subject access request processing.

A reinforcement learning environment for training agents to handle
Data Subject Access Requests (DSARs) under GDPR/UK GDPR compliance.

Three tasks with progressive difficulty:
  - task_easy: Clean consumer request — field-level classification
  - task_medium: Mismatched identity + support ticket redaction
  - task_hard: Weaponised employee DSAR on Slack export
"""

from .client import DSAREnv
from .models import DSARAction, DSARObservation

__all__ = ["DSAREnv", "DSARAction", "DSARObservation"]
