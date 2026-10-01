"""Compatibility entry point for Nsight Compute case captures."""

from .integrations.ncu_profile_case import main

if __name__ == "__main__":
    raise SystemExit(main())
