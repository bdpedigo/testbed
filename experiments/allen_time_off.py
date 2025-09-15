# %%
import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

start_time = datetime.datetime(2025, 8, 29)
print("now =", start_time)

vacation_balance_now = 99.46
base_acrual_rate = 3.08
interval_days = 14
interval = datetime.timedelta(days=interval_days)

end_time = datetime.datetime(2026, 12, 31)

time_offs = {
    # mac bach
    datetime.datetime(2025, 10, 17),
}
# mac ambar wedding
time_offs |= {
    datetime.datetime(2026, 2, 5),
    datetime.datetime(2026, 2, 6),
    datetime.datetime(2026, 2, 9),
    datetime.datetime(2026, 2, 10),
}
# josh wedding placeholder
time_offs |= {
    datetime.datetime(2026, 5, 26),
    datetime.datetime(2026, 5, 27),
    datetime.datetime(2026, 5, 28),
    datetime.datetime(2026, 5, 29),
    datetime.datetime(2026, 6, 1),
    datetime.datetime(2026, 6, 2),
    datetime.datetime(2026, 6, 3),
    datetime.datetime(2026, 6, 4),
    datetime.datetime(2026, 6, 5),
}


rows = []
current_time = start_time
step_interval = datetime.timedelta(days=1)
while current_time < end_time:
    if current_time > datetime.datetime(2026, 9, 21):
        acrual_rate = base_acrual_rate * 1.2
    else:
        acrual_rate = base_acrual_rate
    current_time += step_interval
    if current_time in time_offs:
        vacation_balance_now -= 8.0
    if (current_time - start_time) % interval == datetime.timedelta(0):
        vacation_balance_now += acrual_rate
    rows.append({"date": current_time, "balance": vacation_balance_now})

sns.set_context("talk")
df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"])
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=df, x="date", y="balance", ax=ax)
ax.axhline(180, color="red", linestyle="--")
# %%
