from typing import Optional

import mlflow
from livelossplot import PlotLosses

from src.exp.nb_05b import Callback, AvgStatsCallback, listify


class RecorderMlFlowCallback(Callback):
    def __init__(
        self,
        experiment_name: str,
        run_name: str = "",
        tracking_uri: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.step = 0

    def begin_fit(self):
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        print(f"mlflow tracking uri: {mlflow.get_tracking_uri()}")
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name)

    def after_batch(self):
        if not self.in_train:
            return
        mlflow.log_metric("lr", self.opt.param_groups[-1]["lr"], step=self.step)
        mlflow.log_metric("loss", float(self.loss.detach().cpu()), step=self.step)
        self.step += 1

    def after_fit(self):
        mlflow.end_run()


class AvgStatsMlFlowCallback(AvgStatsCallback):
    def __init__(
        self,
        metrics,
        experiment_name: str,
        run_name: str = "",
        params: Optional[dict] = None,
        tracking_uri: Optional[str] = None,
    ):
        super().__init__(metrics)
        self.metric_names = ["loss"] + [m.__name__ for m in listify(metrics)]
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.step = 0
        self.params = params
        self.set_up_mlflow()

    def set_up_mlflow(self):
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        print(f"mlflow tracking uri: {mlflow.get_tracking_uri()}")
        mlflow.set_experiment(self.experiment_name)

    def begin_fit(self):
        mlflow_runner = mlflow.start_run(run_name=self.run_name)
        self.run.mlflow_uuid = mlflow_runner.info.run_uuid
        if isinstance(self.params, dict):
            for key, val in self.params.items():
                mlflow.log_param(key, val)

    def after_epoch(self):
        super().after_epoch()
        self._log_metrics(self.train_stats.avg_stats, "train")
        self._log_metrics(self.valid_stats.avg_stats, "valid")
        self.step += 1

    def _log_metrics(self, stats, prefix: str):
        for name, val in zip(self.metric_names, stats):
            mlflow.log_metric(f"{prefix}_{name}", float(val), self.step)

    def after_fit(self):
        mlflow.end_run()


class LivelossCallback(AvgStatsCallback):
    def __init__(self, metrics):
        super().__init__(metrics)
        self.liveloss = PlotLosses(skip_first=0)
        self.metricnames = [m.__name__ for m in metrics]
        self.logs = {}

    def begin_epoch(self):
        super().begin_epoch()
        self.logs = {}
        self.iteration = 0

    def after_loss(self):
        super().after_loss()
        if self.in_train:
            self.iteration += 1
            print(
                "\r[%d, %5d] Train_loss: %.3f"
                % (self.epoch + 1, self.iteration, self.loss),
                end="",
            )

    def after_epoch(self):
        super().after_epoch()
        self.logs["loss"] = self.train_stats.avg_stats[0]
        self.logs["val_loss"] = self.valid_stats.avg_stats[0]
        for i, metric in enumerate(self.metricnames):
            self.logs[metric] = self.train_stats.avg_stats[i + 1].item()
            self.logs["val_" + metric] = self.valid_stats.avg_stats[i + 1].item()
        self.liveloss.update(self.logs)
        self.liveloss.draw()
