import os

from transformers import TrainerCallback, is_tensorboard_available
from transformers.integrations import rewrite_logs


class LogFlosCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that adds current flos to every log.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs["total_flos"] = state.total_flos


class TensorBoardFloIndexedCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that sends the logs to `TensorBoard
    <https://www.tensorflow.org/tensorboard>`__.

    Args:
        tb_writer (:obj:`SummaryWriter`, `optional`):
            The writer to use. Will instantiate one if not set.
    """

    def __init__(self, tb_writer=None):
        has_tensorboard = is_tensorboard_available()
        assert (
            has_tensorboard
        ), "TensorBoardCallback requires tensorboard to be installed. Either update your PyTorch version or install tensorboardX."
        if has_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # noqa: F401

                self._SummaryWriter = SummaryWriter
            except ImportError:
                try:
                    from tensorboardX import SummaryWriter

                    self._SummaryWriter = SummaryWriter
                except ImportError:
                    self._SummaryWriter = None
        else:
            self._SummaryWriter = None
        self.tb_writer = tb_writer

    def _init_summary_writer(self, args, log_dir=None):
        log_dir = log_dir or args.logging_dir
        if self._SummaryWriter is not None:
            self.tb_writer = self._SummaryWriter(log_dir=log_dir)

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        log_dir = None

        if state.is_hyper_param_search:
            trial_name = state.trial_name
            if trial_name is not None:
                log_dir = os.path.join(args.logging_dir, trial_name)

        self._init_summary_writer(args, log_dir)

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", args.to_json_string())
            if "model" in kwargs:
                model = kwargs["model"]
                if hasattr(model, "config") and model.config is not None:
                    model_config_json = model.config.to_json_string()
                    self.tb_writer.add_text("model_config", model_config_json)
            # Version of TensorBoard coming from tensorboardX does not have this method.
            if hasattr(self.tb_writer, "add_hparams"):
                self.tb_writer.add_hparams(args.to_sanitized_dict(), metric_dict={})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.tb_writer is None:
            self._init_summary_writer(args)

        if self.tb_writer is not None:
            logs = rewrite_logs(logs)
            self.tb_writer.add_scalar("Conversion/x steps - y flos", state.total_flos, state.global_step)
            self.tb_writer.add_scalar("Conversion/x flos - y steps", state.global_step, state.total_flos)
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(f"Flos/{k}", v, state.total_flos)
                    self.tb_writer.add_scalar(f"Steps/{k}", v, state.global_step)
            self.tb_writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.tb_writer:
            self.tb_writer.close()
