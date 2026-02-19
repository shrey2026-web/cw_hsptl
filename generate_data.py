# generate_data.py
import numpy as np
import pandas as pd

def logistic(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def generate_ed_data(
    n: int = 5000,
    start_date: str = "2025-10-01",
    days: int = 90,
    seed: int = 42,
    out_csv: str = "ed_visits.csv",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Time base ---
    start = pd.Timestamp(start_date)
    # random arrivals across a window
    arrival = start + pd.to_timedelta(rng.uniform(0, days * 24 * 60, size=n), unit="m")
    arrival = arrival.floor("min")

    dow = arrival.dayofweek.values  # 0=Mon..6=Sun
    hour = arrival.hour.values

    # --- Seasonality: volume/crowding patterns ---
    # Busy hours: late afternoon/evening peak, low at night
    hour_peak = (
        0.7 * np.exp(-((hour - 18) / 4.5) ** 2) +   # evening peak
        0.4 * np.exp(-((hour - 10) / 4.0) ** 2) +   # late morning mini-peak
        0.10
    )
    # Weekend slightly busier
    weekend = (dow >= 5).astype(int)
    dow_effect = 1.0 + 0.10 * weekend - 0.05 * (dow == 1)  # Tue slightly lighter

    # Flu-wave anomaly window: days 35-55 of the 90-day period
    day_index = ((arrival - start) / pd.Timedelta(days=1)).astype(int).values
    flu_wave = ((day_index >= 35) & (day_index <= 55)).astype(int)

    # bed occupancy: correlated with busy hours + weekend + flu wave + noise
    bed_occ = (
        55
        + 30 * (hour_peak / hour_peak.max())
        + 6 * weekend
        + 10 * flu_wave
        + rng.normal(0, 6, size=n)
    )
    bed_occ = np.clip(bed_occ, 35, 98).round(1)

    # --- Patient attributes ---
    age_group = rng.choice(
        ["0-17", "18-34", "35-49", "50-64", "65+"],
        size=n,
        p=[0.10, 0.30, 0.22, 0.20, 0.18],
    )

    # triage level (ESI 1-5): more 3s/4s, fewer 1s
    triage = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.03, 0.12, 0.45, 0.30, 0.10])

    arrival_mode = np.where((triage <= 2) & (rng.random(n) < 0.65), "Ambulance", "Walk-in")
    arrival_mode = np.where(rng.random(n) < 0.08, "Ambulance", arrival_mode)  # some low acuity by EMS

    pod = rng.choice(["Pod A", "Pod B", "Pod C"], size=n, p=[0.34, 0.33, 0.33])

    # chief complaints shift during flu wave
    base_complaints = np.array(["Chest Pain", "Abdominal Pain", "Injury", "Fever/Resp", "Headache", "Other"])
    # baseline probs
    p_base = np.array([0.10, 0.18, 0.22, 0.16, 0.08, 0.26])
    # during flu wave, bump Fever/Resp + Other, reduce Injury a bit
    p_flu = np.array([0.09, 0.16, 0.16, 0.28, 0.07, 0.24])

    complaint = []
    for i in range(n):
        probs = p_flu if flu_wave[i] == 1 else p_base
        complaint.append(rng.choice(base_complaints, p=probs))
    complaint = np.array(complaint)

    # orders: depend on triage + complaint
    labs_prob = (
        0.20
        + 0.15 * (triage <= 2)
        + 0.10 * (triage == 3)
        + 0.08 * (complaint == "Chest Pain")
        + 0.06 * (complaint == "Fever/Resp")
    )
    imaging_prob = (
        0.12
        + 0.14 * (complaint == "Injury")
        + 0.10 * (complaint == "Chest Pain")
        + 0.06 * (triage <= 2)
    )

    labs_ordered = (rng.random(n) < np.clip(labs_prob, 0, 0.95)).astype(int)
    imaging_ordered = (rng.random(n) < np.clip(imaging_prob, 0, 0.90)).astype(int)

    # --- Operational outcomes ---
    # door-to-provider: increases with occupancy, also slightly higher during flu wave
    # but high acuity gets faster provider time.
    dtp_mean = (
        18
        + 0.55 * (bed_occ - 60)
        + 6 * flu_wave
        - 8 * (triage <= 2)
        - 3 * (triage == 3)
        + rng.normal(0, 8, size=n)
    )
    door_to_provider_min = np.clip(dtp_mean, 2, 240).round(0).astype(int)

    # LOS: base + crowding + orders + admissions tendency
    base_los = 110 + 0.45 * (bed_occ - 60) + rng.normal(0, 25, size=n)
    los = base_los + 55 * labs_ordered + 65 * imaging_ordered
    # high acuity generally longer LOS (workup/admission)
    los += 70 * (triage <= 2) + 15 * (triage == 3)
    # flu wave adds some extra throughput burden
    los += 15 * flu_wave
    length_of_stay_min = np.clip(los, 25, 900).round(0).astype(int)

    # disposition: admission more likely for high acuity + chest pain/older
    age65 = (age_group == "65+").astype(int)
    admit_score = (
        -2.0
        + 1.5 * (triage <= 2)
        + 0.7 * (triage == 3)
        + 0.6 * (complaint == "Chest Pain")
        + 0.5 * (complaint == "Fever/Resp")
        + 0.6 * age65
        + 0.02 * (bed_occ - 70)
    )
    admit_prob = logistic(admit_score)

    # LWBS: more likely if long wait and high crowding, less likely high acuity
    lwbs_score = (
        -4.0
        + 0.018 * door_to_provider_min
        + 0.035 * (bed_occ - 70)
        - 1.2 * (triage <= 2)
        - 0.4 * (triage == 3)
        + 0.25 * flu_wave
    )
    lwbs_prob = np.clip(logistic(lwbs_score), 0, 0.40)

    r = rng.random(n)
    disposition = np.where(r < lwbs_prob, "Left Without Being Seen", "Discharged")
    # if not LWBS, admit with probability admit_prob
    r2 = rng.random(n)
    disposition = np.where((disposition != "Left Without Being Seen") & (r2 < admit_prob), "Admitted", disposition)

    df = pd.DataFrame({
        "visit_id": np.arange(1, n + 1),
        "arrival_datetime": arrival.astype("datetime64[ns]"),
        "day_of_week": arrival.day_name(),
        "hour": hour,
        "triage_level": triage,
        "chief_complaint": complaint,
        "age_group": age_group,
        "arrival_mode": arrival_mode,
        "pod": pod,
        "labs_ordered": labs_ordered,
        "imaging_ordered": imaging_ordered,
        "bed_occupancy_pct": bed_occ,
        "door_to_provider_min": door_to_provider_min,
        "length_of_stay_min": length_of_stay_min,
        "disposition": disposition,
        "flu_wave_flag": flu_wave,
    })

    # A couple of convenient derived fields for dashboard
    df["arrival_date"] = pd.to_datetime(df["arrival_datetime"]).dt.date
    df["is_weekend"] = (pd.to_datetime(df["arrival_datetime"]).dt.dayofweek >= 5).astype(int)

    df.to_csv(out_csv, index=False)
    return df

if __name__ == "__main__":
    df = generate_ed_data()
    print("Saved:", df.shape, "to ed_visits.csv")
    print(df.head(3))
