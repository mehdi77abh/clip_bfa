"""
main.py — Entry point for the hybrid bit-flip attack pipeline.
"""

from config import CFG
from pipeline import AttackPipeline


def main():
    cfg = CFG()
    AttackPipeline(cfg).run()


if __name__ == "__main__":
    main()