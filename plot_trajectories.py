import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    try:
        df = pd.read_csv("face_logs.csv")
    except FileNotFoundError:
        print("face_logs.csv not found. Run main.py first.")
        return

    if df.empty:
        print("Log file is empty, nothing to plot.")
        return

    plt.figure(figsize=(7, 6))
    for fid, sub in df.groupby("face_id"):
        x = sub["smooth_cx"] if "smooth_cx" in sub.columns else sub["cx"]
        y = sub["smooth_cy"] if "smooth_cy" in sub.columns else sub["cy"]
        plt.plot(x, y, marker=".", linestyle="-", label=f"Face {fid}")
    plt.gca().invert_yaxis()
    plt.xlabel("x center (pixels)")
    plt.ylabel("y center (pixels)")
    plt.title("Face Trajectories")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if "reaction_time_ms" in df.columns:
        rt = pd.to_numeric(df["reaction_time_ms"], errors="coerce").dropna()
        if not rt.empty:
            plt.figure(figsize=(6, 4))
            plt.hist(rt, bins=20)
            plt.xlabel("Reaction time (ms)")
            plt.ylabel("Count")
            plt.title("Reaction Time Distribution")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
