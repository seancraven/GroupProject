try:
    import wandb
except ImportError:
    pass


class ReporterMixin:
    @property
    def _verbosity(self):
        return getattr(self, "verbosity", 1)

    def debug(self, *args, **kwargs) -> None:
        if self._verbosity >= 2:
            print(*args, **kwargs)

    def info(self, *args, **kwargs) -> None:
        if self._verbosity >= 1:
            print(*args, **kwargs)

    @staticmethod
    def wandb_init(*args, **kwargs) -> None:
        try:
            wandb.init(*args, **kwargs)  # type: ignore
        except:
            pass

    @staticmethod
    def _wandb_log_no_print(*args, **kwargs) -> None:
        try:
            wandb.log(*args, **kwargs)  # type: ignore
        except:
            pass

    def wandb_log(self, log_dict: dict) -> None:
        log_string = [f"{k}: {v:.4f}" for k, v in log_dict.items()]
        log_string = "\t" + " | ".join(log_string)  # type: ignore
        self.info(log_string)
        return self._wandb_log_no_print(log_dict)

    def wandb_log_named(self, log_dict: dict, name: str) -> None:
        named_log_dict = {f"{name} | {k}": v for k, v in log_dict.items()}
        log_strings = [f"{k}: {v:.4f}" for k, v in log_dict.items()]
        log_string = "\t" + " | ".join(log_strings)
        self.info(log_string)
        return self._wandb_log_no_print(named_log_dict)
