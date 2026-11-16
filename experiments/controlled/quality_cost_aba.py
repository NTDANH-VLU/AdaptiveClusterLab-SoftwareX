"""Backward-compatible script shim for the installable AdaptiveClusterLab controller.

Real implementation: adaptiveclusterlab.controller
"""
from adaptiveclusterlab.controller import *  # noqa: F401,F403

if __name__ == "__main__":
    main()
