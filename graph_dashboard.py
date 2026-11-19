from typing import get_args
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from policy_classification import PolicyClassification

def graph_dashboard(df, datafile):
    df = pd.read_csv(datafile)

    df["target_sector"] = df["target_sector"].replace({"General Environment": "Other"})

    MECHANISMS = list(get_args(PolicyClassification.model_fields["policy_mechanism"].annotation))

    SECTORS = list(get_args(PolicyClassification.model_fields["target_sector"].annotation))

    ALIGNMENTS = list(get_args(PolicyClassification.model_fields["advocacy_alignment"].annotation))

    STATUS_MAP = {
        1: "Introduced",
        2: "Engrossed",
        3: "Enrolled",
        4: "Enacted",
        5: "Vetoed",
        6: "Failed",
    }

    STATUS_ORDER = list(STATUS_MAP.values())
    df["status_label"] = df["status"].map(STATUS_MAP)

    ALIGNMENT_COLORS = {
        "Strict Regulation": "#2b8cbe",
        "Neutral / Procedural": "#a6bddb",
        "Industry Relief / Exemption": "#fdbb84",
    }

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(18.5, 13.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.58], hspace=0.35, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax_tbl = fig.add_subplot(gs[2, :])

    # fig 1
    sector_counts = df["target_sector"].value_counts().reindex(SECTORS, fill_value=0)
    sector_counts = sector_counts[sector_counts > 0]

    pie_colors = sns.color_palette("crest", len(sector_counts))
    total_bills = sum(sector_counts.values)
    pie_labels = [
        f"{sector} ({count/total_bills*100:.1f}%)"
        for sector, count in zip(sector_counts.index, sector_counts.values)
    ]

    wedges, texts = ax1.pie(
        sector_counts.values,
        labels=pie_labels,
        startangle=227,
        colors=pie_colors,
        labeldistance=1.1,
        wedgeprops=dict(edgecolor="white", linewidth=1.2),
    )
    plt.setp(texts, size=8.5)
    ax1.set_title("Targeted Conservation Sectors", fontweight="bold", fontsize=11, pad=14)


    # fig 2
    ct = pd.crosstab(df["policy_mechanism"], df["advocacy_alignment"]).reindex(
        index=MECHANISMS, columns=ALIGNMENTS, fill_value=0
    )
    ct.plot(
        kind="barh",
        stacked=True,
        ax=ax2,
        color=["#1f83b4", "#ffae6b", "#a6bddb"],
        edgecolor="black",
        linewidth=0.6,
    )
    ax2.invert_yaxis()
    ax2.set_title("Policy Mechanism by Advocacy Alignment", fontsize=10)
    ax2.set_xlabel("Number of Bills")
    ax2.set_ylabel("")
    ax2.legend(title="Alignment", fontsize=9, loc="lower right")

    # fig 3
    sns.scatterplot(
        data=df,
        x="days_active",
        y="action_count",
        hue="status_label",
        style="status_label",
        hue_order=STATUS_ORDER,
        style_order=STATUS_ORDER,
        s=75,
        alpha=0.85,
        ax=ax3,
    )
    ax3.set_title("Legislative Velocity: Days Active vs. Action Count", fontweight="bold", fontsize=11)
    ax3.set_xlabel("Days Active (Introduction to Latest Action)")
    ax3.set_ylabel("Total Legislative Actions")
    ax3.legend(title="Bill Status", fontsize=9, loc="upper right")

    # fig 4
    top_states = df["state"].value_counts().head(12)
    sns.barplot(
        x=top_states.index,
        y=top_states.values,
        hue=top_states.index,
        palette="viridis",
        legend=False,
        ax=ax4,
    )
    ax4.set_title("Jurisdictional Distribution of Sampled Conservation Bills", fontweight="bold", fontsize=11)
    ax4.set_xlabel("Jurisdiction (State / Federal)")
    ax4.set_ylabel("Bill Count")
    ax4.set_ylim(0, top_states.max() * 1.15)
    for i, v in enumerate(top_states.values):
        ax4.text(i, v + 0.8, str(v), ha="center", fontweight="bold", fontsize=10)

    # fig 5
    ax_tbl.axis("off")
    top_clusters = (
        df.groupby("policy_cluster")
        .agg(
            Bills=("bill_id", "count"),
            Jurisdictions=("state", "nunique"),
            States=("state", lambda s: ", ".join(sorted(s.unique())[:5]) + ("..." if s.nunique() > 5 else "")),
            Dominant_Mechanism=("policy_mechanism", lambda s: s.mode().iloc[0]),
            Representative_Bill=("title", lambda s: s.iloc[0][:125] + ("..." if len(s.iloc[0]) > 125 else "")),
        )
        .sort_values(["Bills", "Jurisdictions"], ascending=False)
        .head(5)
        .reset_index()
    )
    top_clusters["Cluster"] = top_clusters["policy_cluster"].apply(lambda c: f"#{c}")
    table_df = top_clusters[["Cluster", "Bills", "Jurisdictions", "States", "Dominant_Mechanism", "Representative_Bill"]]
    table_df.columns = ["Cluster", "Bills", "# of States", "Jurisdictions", "Dominant Mechanism", "Representative Bill Desc"]

    tbl = ax_tbl.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="left",
        colWidths=[0.04, 0.03, 0.06, 0.10, 0.18, 0.59],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    for (row, col), cell in tbl.get_celld().items():
        if col >= 3:
            cell.PAD = 0.025
        cell.set_edgecolor("#d0d7de")
        if row == 0:
            cell.set_facecolor("#2b3e50")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row % 2 == 1:
            cell.set_facecolor("#f8f9fa")

    ax_tbl.set_title(
        f"Top Semantic Policy Clusters ({df['policy_cluster'].nunique()} Total Clusters Across {len(df)} Bills — Interstate Diffusion)",
        fontweight="bold",
        fontsize=11,
        pad=8,
    )

    fig.suptitle(
        f"Conservation Legislative Regulatory Intelligence Dashboard (n = {len(df)} Bills)",
        fontsize=14,
        fontweight="bold",
        y=0.96,
    )
    plt.savefig("conservation_legislative_dashboard.png", dpi=200, bbox_inches="tight")
    plt.show()