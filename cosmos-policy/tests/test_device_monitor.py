from cosmos_policy._src.predict2.callbacks import device_monitor


def test_log_prof_data_can_skip_wandb_table_but_keep_scalar_summaries(monkeypatch):
    logged = []

    monkeypatch.setattr(device_monitor.wandb, "run", object())
    monkeypatch.setattr(
        device_monitor.wandb,
        "Table",
        lambda **_: (_ for _ in ()).throw(AssertionError("table artifact should be disabled")),
    )
    monkeypatch.setattr(device_monitor.wandb, "log", lambda payload, step: logged.append((payload, step)))

    _, summary = device_monitor.log_prof_data(
        [
            {"util": 25.0, "memory": 10.0},
            {"util": 75.0, "memory": 14.0},
        ],
        iteration=10,
        log_wandb_table=False,
    )

    assert summary.loc["util"].to_dict() == {"Avg": 50.0, "Max": 75.0, "Min": 25.0}
    assert logged == [
        (
            {
                "DeviceMonitor/min_util": 25.0,
                "DeviceMonitor/max_util": 75.0,
                "DeviceMonitor/avg_util": 50.0,
                "DeviceMonitor/min_memory": 10.0,
                "DeviceMonitor/max_memory": 14.0,
                "DeviceMonitor/avg_memory": 12.0,
            },
            10,
        )
    ]
