"""Locally maintained Tri-Domain (time/freq/space) EEG classifier."""

from .model import TriDomainClassifier, TriDomainEncoder, build_model

__all__ = ["TriDomainClassifier", "TriDomainEncoder", "build_model"]
